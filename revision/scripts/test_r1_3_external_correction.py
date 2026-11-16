"""Focused [decision record] guards and mathematically checkable saved-ranking fixtures."""

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import r1_3_external_correction as c


class ClusterTests(unittest.TestCase):
    def test_distance_roundoff_only_does_not_relax_other_outcomes(self):
        expected = pd.DataFrame({"query": ["q"], "true_label": ["WT"],
                                 "mean_top5_distance": [3819.037959660297],
                                 "top1_distance": [4000.0], "hit_k5": [1.]})
        derived = expected.copy()
        derived.loc[0, "mean_top5_distance"] -= 1.3642420526593924e-12
        c.compare_per_query_frames(expected, derived, "fixture")
        derived.loc[0, "hit_k5"] += 2e-12
        with self.assertRaisesRegex(ValueError, "numeric outcome differs: hit_k5"):
            c.compare_per_query_frames(expected, derived, "fixture")
        derived.loc[0, "hit_k5"] = 1.
        derived.loc[0, "true_label"] = "NB"
        with self.assertRaisesRegex(ValueError, "categorical outcome differs: true_label"):
            c.compare_per_query_frames(expected, derived, "fixture")

    def test_whole_patient_multiplicity_and_variable_micro_denominator(self):
        pop = pd.DataFrame({
            "slide_id": ["a1", "a2", "a3", "b", "c", "d"],
            "tumor_group": ["WT"] * 4 + ["NB"] * 2,
            "patient_id": ["a"] * 3 + ["b", "c", "d"]})
        w = c.cluster_weights(pop, "patient_id", b=300, seed=123)
        np.testing.assert_array_equal(w[:, 0], w[:, 1])
        np.testing.assert_array_equal(w[:, 1], w[:, 2])
        np.testing.assert_array_equal(w[:, 0] + w[:, 3], 2)
        np.testing.assert_array_equal(w[:, 4] + w[:, 5], 2)
        self.assertGreater(np.unique(w.sum(axis=1)).size, 1)
        # Draw both 3-slide patient copies: four of six slides succeed, not
        # four of the original six slides and not two of four patient draws.
        v = np.array([1, 1, 1, 0, 0, 0])
        ix = np.flatnonzero(w[:, 0] == 2)[0]
        self.assertAlmostEqual((w[ix] @ v) / w[ix].sum(), 6 / 8)

    def test_fixed_macro_even_with_singleton_stratum(self):
        pop = pd.DataFrame({"slide_id": ["a", "b", "c", "d"],
                            "tumor_group": ["WT", "WT", "WT", "NB"],
                            "patient_id": ["a", "b", "c", "d"]})
        w = c.cluster_weights(pop, "patient_id", b=80, seed=123)
        self.assertTrue(np.all(w[:, 3] == 1))
        np.testing.assert_array_equal(w.sum(axis=1), np.full(80, 4))
        classes = [(w[:, :3] @ np.array([1., 0., 0.])) /
                   w[:, :3].sum(axis=1), np.ones(80)]
        fixed_macro = np.mean(classes, axis=0)
        self.assertTrue(np.all(fixed_macro >= .5))
        self.assertTrue(np.all(fixed_macro <= 1))
        self.assertTrue(np.any(fixed_macro != (w @ np.array([1., 0., 0., 1.])) / 4))

    def test_reject_missing_duplicate_and_cross_class_patient(self):
        pop = pd.DataFrame({"slide_id": ["a", "b"],
                            "tumor_group": ["WT", "NB"],
                            "patient_id": ["same", "same"]})
        with self.assertRaisesRegex(ValueError, "Cross-class"):
            c.cluster_weights(pop, "patient_id", b=2)
        pop.loc[1, "patient_id"] = ""
        with self.assertRaisesRegex(ValueError, "Missing/blank"):
            c.cluster_weights(pop, "patient_id", b=2)
        pop.loc[1, "patient_id"] = None
        with self.assertRaisesRegex(ValueError, "Missing/blank"):
            c.cluster_weights(pop, "patient_id", b=2)
        pop.loc[1, "patient_id"] = "different"
        pop.loc[1, "slide_id"] = "a"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            c.cluster_weights(pop, "patient_id", b=2)

    def test_shared_weights_exact_paired_difference_and_replayed_metrics(self):
        pop = pd.DataFrame({"slide_id": ["a", "b", "c"],
                            "tumor_group": ["WT", "WT", "NB"],
                            "cluster_id": ["a", "b", "c"]})
        ranked_base = np.array([
            ["WT", "NB", "NB", "NB", "NB", "NB", "NB", "NB", "NB", "NB"],
            ["NB"] * 10, ["NB"] * 10], dtype=object)
        ranked_new = ranked_base.copy()
        ranked_new[1, 0] = "WT"
        ranked_new[0, 0] = "NB"
        cfg = {"library_class_counts": {"WT": 2, "NB": 10},
               "library_classes": ["NB", "WT"]}
        w = c.cluster_weights(pop.rename(columns={"cluster_id": "patient"}),
                              "patient", b=50)

        def make(labs):
            ids = np.array([[f"n{i}_{j}" for j in range(10)] for i in range(3)])
            q = c.external_cli.per_query_frame(
                pop.slide_id.tolist(), pop.tumor_group.tolist(), ids, labs,
                np.tile(np.arange(10, dtype=float), (3, 1)), c.K, 10)
            q.attrs["ranked_labels"] = labs
            return q

        old_q, new_q = make(ranked_base), make(ranked_new)
        old, od = c.score(pop, old_q, cfg, w)
        new, nd = c.score(pop, new_q, cfg, w)
        expected = c.metrics.compute_all(ranked_base, pop.tumor_group, c.K,
                                         n_relevant=[2, 2, 10])
        self.assertEqual(old["point_estimates"], expected)
        self.assertAlmostEqual(old["point_estimates"]["macro_hit_at_5"], .75)
        self.assertAlmostEqual(old["point_estimates"]["micro_precision_at_5"], 6 / 15)
        np.testing.assert_allclose(nd["micro_hit_at_5"] - od["micro_hit_at_5"],
                                   (w @ (new_q.hit_k5 - old_q.hit_k5).to_numpy()) /
                                   w.sum(axis=1), atol=1e-15)
        self.assertEqual(new["confidence_intervals_95"]["macro_hit_at_5"]["B_effective"], 50)
        self.assertEqual(len(new["per_class_primary"]), 2)
        self.assertEqual(set(new["observed_classes"]), {"WT", "NB"})
        self.assertEqual(new["confidence_intervals_95"]["per_class_NB_hit_at_5"]["lo"], 1)
        a = {"report": new, "draws": nd, "config": cfg}
        baseline = {"report": old, "draws": od, "config": cfg}
        comparison = c.paired(pop, a, baseline, new_q, old_q,
                              "stv", "main_696", "sampler", "whole_slide_ot")
        self.assertEqual(comparison["comparison_id"],
                         "stv_main_696_sampler_vs_whole_slide_ot")
        self.assertEqual(comparison["alternative_configuration"], "sampler")
        self.assertEqual(comparison["baseline_configuration"], "whole_slide_ot")
        self.assertEqual(comparison["hit_and_vote_fixed_broken"]["hit_at_5"]["fixed"], 1)
        self.assertEqual(comparison["hit_and_vote_fixed_broken"]["hit_at_5"]["broken"], 1)
        self.assertAlmostEqual(
            comparison["primary_and_vote_differences"]["micro_hit_at_5"]["difference"],
            new["point_estimates"]["micro_hit_at_5"] -
            old["point_estimates"]["micro_hit_at_5"])

    def test_paired_output_identities_and_cohort_specific_design(self):
        expected = {
            ("stv", "main_696"): {"median_pooling", "sampler", "yottixel"},
            ("stv", "query_only_696"): {"query_only"},
            ("stv", "both_sided_688"): {"both_sided"},
            ("kidrare", "amended_1581"): {"median_pooling", "sampler", "yottixel"},
            ("kidrare", "legacy_1227"): {"median_pooling", "sampler", "yottixel"},
            ("kidrare", "query_only_1226"): {"query_only"},
            ("kidrare", "both_sided_1176"): {"both_sided"},
        }
        expected_triples = {(cohort, population, alternative)
                            for (cohort, population), alternatives in expected.items()
                            for alternative in alternatives}
        self.assertEqual(c.EXPECTED_PAIRS, expected_triples)
        pop = pd.DataFrame({"slide_id": ["x"], "tumor_group": ["WT"],
                            "cluster_id": ["p"]})
        q = pd.DataFrame({"query": ["x"], "true_label": ["WT"], "hit_k5": [1],
                          "generous_correct_k5": [1]})
        cfg = {"library_classes": ["WT"]}
        report = {"point_estimates": {key: .5 for key in c.PRIMARY},
                  "voting_k5": {"generous": {"accuracy": 1.0}},
                  "macro_generous_accuracy_k5": 1.0}
        draw_keys = (*c.PRIMARY, "micro_generous_accuracy_k5",
                     "macro_generous_accuracy_k5")
        entry = {"report": report, "draws": {key: np.array([.5, 1.0])
                                             for key in draw_keys}, "config": cfg}
        rows = [c.paired(pop, entry, entry, q, q, cohort, population,
                         alternative, "whole_slide_ot")
                for (cohort, population), alternatives in expected.items()
                for alternative in alternatives]
        self.assertEqual(len(rows), len(expected_triples))
        self.assertEqual({(row["cohort"], row["population"],
                           row["alternative_configuration"]) for row in rows},
                         expected_triples)
        self.assertEqual({row["comparison_id"] for row in rows},
                         {f"{cohort}_{population}_{alternative}_vs_whole_slide_ot"
                          for cohort, population, alternative in expected_triples})
        self.assertTrue(all(row["baseline_configuration"] == "whole_slide_ot"
                            for row in rows))
        self.assertEqual(len({row["comparison_id"] for row in rows}), len(rows))
        w = np.ones((2, 1), dtype=np.int16)
        stv = c.bootstrap_description("stv", w, ["WT"])
        kidrare = c.bootstrap_description("kidrare", w, ["WT"])
        self.assertEqual(stv["resampling_unit"], "supplied ST_ patient ID")
        self.assertIn("all eligible slides per drawn patient", stv["design"])
        self.assertEqual(kidrare["resampling_unit"], "slide (nominal)")
        self.assertIn("Class-stratified slide resampling", kidrare["design"])
        self.assertNotIn("per drawn patient", kidrare["design"])
        self.assertEqual(stv["fixed_observed_classes"], kidrare["fixed_observed_classes"])


class InputGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = c.ROOT / "prosper_revision_output"
        cls.paths = c.sources(cls.out)

    def test_explicit_local_output_and_nonempty_refusal(self):
        with patch.dict(os.environ, {"PROSPER_OUT": ""}):
            with self.assertRaisesRegex(ValueError, "explicitly"):
                c.out_root()
        with patch.dict(os.environ, {"PROSPER_OUT": str(Path.home())}):
            with self.assertRaisesRegex(ValueError, "repository-local"):
                c.out_root()
        with self.assertRaisesRegex(ValueError, "Refusing nonempty"):
            c.refuse_nonempty(c.ROOT / "scripts")

    def test_reject_duplicate_unknown_and_conflicting_rank_labels(self):
        q = pd.DataFrame({"query": ["a", "b"]})
        lib = pd.Series(["WT"] * 10 + ["NB"] * 10,
                        index=[f"x{i}" for i in range(20)])
        r = pd.DataFrame({
            "query": ["a"] * 10 + ["b"] * 10,
            "rank": list(range(1, 11)) * 2,
            "neighbour": list(lib.index),
            "neighbour_label": list(lib.to_numpy()),
            "distance": [float(i) for i in range(20)]})
        c.validate_ranks(r, q, set(lib.index), lib, "fixture")
        for column, value in (("rank", 2), ("neighbour", "x1"),
                              ("neighbour", "unknown"),
                              ("neighbour_label", "WRONG"),
                              ("distance", np.nan)):
            bad = r.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column, value=value):
                with self.assertRaisesRegex(ValueError, "rank depth/uniqueness"):
                    c.validate_ranks(bad, q, set(lib.index), lib, "fixture")

    def test_actual_stv_and_kidrare_source_guards(self):
        c.validate_library_filter(self.paths, c.csv(self.paths["comet_metadata"]))
        stv = c.stv_manifest(self.paths)
        self.assertEqual((len(stv), int(stv.main_eligible.sum()),
                          int(stv.both_eligible.sum())), (782, 696, 688))
        ranked = pd.read_csv(self.paths["kidrare_whole_slide_ot_ranked"],
                             usecols=["query"], dtype=str)
        kid = c.kid_manifest(self.paths, set(ranked["query"]))
        self.assertEqual((len(kid), int(kid.pathpt_only_new_label.sum()),
                          int(kid.amended_eligible.sum()),
                          int((kid.legacy_eligible & kid.slide_id.ne(c.ZERO_LIVER)).sum())),
                         (2331, 357, 1581, 1226))
        self.assertEqual(
            int((kid.legacy_eligible & kid.slide_id.ne(c.ZERO_LIVER) &
                 kid.tumor_group.ne("NORMAL")).sum()), 1176)
        self.assertFalse(kid.loc[kid.pathpt_only_new_label, "keep_subtype"].eq("Normal").any())
        self.assertEqual(stv.loc[stv.both_eligible, "patient_id"].nunique(), 346)

    def test_frozen_input_and_unchanged_ccdi_bindings(self):
        complete = json.loads(self.paths["ccdi_complete"].read_text())
        self.assertEqual(complete["provenance_sha256"], c.R1_2_SHA)
        self.assertEqual(c.sha(self.paths["ccdi_provenance"]), c.R1_2_SHA)
        source = json.loads(self.paths["ccdi_provenance"].read_text())["input_sha256"]
        for current, prior in (("comet_metadata", "comet_metadata"),
                               ("metrics_helper", "shared_metrics"),
                               ("external_helper", "shared_external_cli")):
            self.assertEqual(c.sha(self.paths[current]), source[prior]["sha256"])
            self.assertEqual(self.paths[current].resolve(),
                             Path(source[prior]["path"]).resolve())
        self.assertEqual(c.sha(self.paths["stv_source"]), c.STV_SOURCE_SHA)
        self.assertEqual(c.sha(self.paths["pathpt"]), c.PATHPT_SHA)

    def test_conflicting_keep_source_fails(self):
        manifest = c.csv(self.paths["kid_manifest"])
        pt = json.loads(self.paths["pathpt"].read_text())
        keep = json.loads(self.paths["keep_wt"].read_text())
        first = next(iter(keep))
        keep[first] = "Tumor" if keep[first] == "Normal" else "Normal"
        with self.assertRaisesRegex(ValueError, "KEEP"):
            c.pathpt_kidney(pt, manifest, keep)

    def test_all_saved_rank_source_legacy_and_filter_fidelity(self):
        stv = c.stv_manifest(self.paths)
        kid_all = c.csv(self.paths["kidrare_whole_slide_ot_per_query"])
        kid = c.kid_manifest(self.paths, set(kid_all["query"]))
        metadata = c.csv(self.paths["comet_metadata"])
        for cohort, d in (("stv", stv), ("kidrare", kid)):
            labels = dict(zip(d.loc[d.processed, "slide_id"],
                              d.loc[d.processed, "tumor_group" if cohort == "stv"
                                    else "original_tumor_group"]))
            legacy = set(d.loc[d.main_eligible if cohort == "stv" else
                               d.legacy_eligible, "slide_id"])
            for name in c.METHODS:
                q, r, source = c.replay(
                    name, cohort, self.paths, metadata, labels, legacy,
                    776 if cohort == "stv" else 2322)
                if cohort == "stv" and name == "whole_slide_ot":
                    self.assertAlmostEqual(
                        source["point_estimates"]["macro_hit_at_5"], .908577, places=5)
                if cohort == "kidrare":
                    lq, lr, old = c.replay("legacy_" + name, cohort, self.paths,
                                           metadata, {x: labels[x] for x in legacy},
                                           legacy, 1227)
                    self.assertEqual(source["point_estimates"], old["point_estimates"])
                    self.assertTrue(lq.equals(q.loc[lq["query"]]))
                    self.assertTrue(lr.sort_index().equals(
                        r.loc[pd.MultiIndex.from_product([
                            lq["query"], range(1, 11)])].sort_index()))
            for arm in ("query_only", "both_sided"):
                evaluated = legacy if cohort == "stv" else legacy - {c.ZERO_LIVER}
                if arm == "both_sided":
                    evaluated = {i for i in evaluated if labels[i] != "NORMAL"}
                filtered_labels = labels if cohort == "stv" else {
                    i: labels[i] for i in legacy - {c.ZERO_LIVER}}
                q, r, source = c.replay(
                    arm, cohort, self.paths, metadata, filtered_labels,
                    evaluated, 776 if cohort == "stv" else 1226)
                self.assertEqual(source["n_in_library_queries"], len(evaluated))
                if cohort == "kidrare":
                    self.assertFalse(set(q.index) &
                                     set(kid.loc[kid.pathpt_only_new_label, "slide_id"]))
                    self.assertNotIn(c.ZERO_LIVER, q.index)
                if arm == "both_sided":
                    self.assertEqual(source["config"]["n_library"], 2603)

    def test_real_small_draws_and_new_wt_truth_recomputed(self):
        metadata = c.csv(self.paths["comet_metadata"])
        stv = c.stv_manifest(self.paths)
        kid_all = c.csv(self.paths["kidrare_whole_slide_ot_per_query"])
        kid = c.kid_manifest(self.paths, set(kid_all["query"]))
        for cohort, d in (("stv", stv), ("kidrare", kid)):
            labels = dict(zip(d.loc[d.processed, "slide_id"],
                              d.loc[d.processed, "tumor_group" if cohort == "stv"
                                    else "original_tumor_group"]))
            old_ids = set(d.loc[d.main_eligible if cohort == "stv" else
                                d.legacy_eligible, "slide_id"])
            q, ranks, source = c.replay(
                "whole_slide_ot", cohort, self.paths, metadata, labels,
                old_ids, 776 if cohort == "stv" else 2322)
            pop = d.loc[d.main_eligible if cohort == "stv" else
                        d.amended_eligible].copy()
            if cohort == "stv":
                pop["cluster_id"] = pop.patient_id
            else:
                pop["tumor_group"] = pop.corrected_tumor_group
                pop["cluster_id"] = pop.slide_id
            revised, rr = c.derive(pop, q, ranks, source["config"])
            self.assertEqual(len(rr), len(pop) * 10)
            w = c.cluster_weights(pop, "cluster_id", b=12, seed=123)
            report, draws = c.score(pop, revised, source["config"], w)
            self.assertEqual(report["confidence_intervals_95"]["macro_precision_at_5"][
                "B_effective"], 12)
            self.assertEqual(len(draws["micro_hit_at_5"]), 12)
            if cohort == "stv":
                self.assertGreater(int(w.sum(axis=1).max()), int(w.sum(axis=1).min()))
            else:
                changed = set(d.loc[d.pathpt_only_new_label & d.processed, "slide_id"])
                new = revised.set_index("query").loc[list(changed)]
                old = q.loc[list(changed)]
                self.assertEqual(len(changed), 354)
                self.assertTrue(new.true_label.eq("WT").all())
                self.assertTrue(old.true_label.eq("WT_UNVERIFIED").all())
                from_labels = rr.loc[rr["rank"].le(5)].groupby("query").neighbour_label.agg(
                    lambda x: "WT" in set(x))
                np.testing.assert_array_equal(new.hit_k5.to_numpy(dtype=bool),
                                              from_labels.loc[new.index].to_numpy())

    def test_measured_legacy_sampler_distance_roundoff(self):
        metadata = c.csv(self.paths["comet_metadata"])
        processed = set(c.csv(self.paths["kidrare_whole_slide_ot_per_query"])["query"])
        kid = c.kid_manifest(self.paths, processed)
        legacy = kid.loc[kid.legacy_eligible].copy()
        legacy["tumor_group"] = legacy.original_tumor_group
        labels = dict(zip(kid.loc[kid.processed, "slide_id"],
                          kid.loc[kid.processed, "original_tumor_group"]))
        q, ranked, source = c.replay("sampler", "kidrare", self.paths, metadata,
                                     labels, set(legacy.slide_id), 2322)
        derived, _ = c.derive(legacy, q, ranked, source["config"])
        original = q.loc[legacy.slide_id].reset_index(drop=True)
        delta = np.abs(original.mean_top5_distance.to_numpy() -
                       derived.mean_top5_distance.to_numpy())
        self.assertEqual(int((delta > 1e-12).sum()), 2)
        self.assertAlmostEqual(float(delta.max()), 1.3642420526593924e-12,
                               delta=1e-14)
        c.compare_per_query_frames(original, derived, "KidRare legacy sampler")

    def test_end_to_end_small_bootstrap_isolated_output_self_cleans(self):
        target = self.out / f"r1_3_correction_test_{os.getpid()}"
        self.assertFalse(target.exists(), "Refusing to touch a pre-existing test destination")
        production = self.out / c.DEST
        failed = self.out / "r1_3_external_correction_20260930_failed_324493978"

        def snapshot(directory):
            if not directory.exists():
                return (False, {})
            self.assertTrue(directory.is_dir())
            return (True, {entry.name: c.sha(entry) if entry.is_file() else "directory"
                           for entry in directory.iterdir()})

        before = {path: snapshot(path) for path in (production, failed)}
        try:
            with patch.dict(os.environ, {"PROSPER_OUT": str(self.out)}), \
                    patch.object(c, "DEST", target.name), patch.object(c, "B", 8):
                c.run()
                with self.assertRaisesRegex(ValueError, "Refusing nonempty"):
                    c.run()
            complete = json.loads((target / "COMPLETE.json").read_text())
            self.assertEqual((complete["status"], complete["n_reports"],
                              complete["n_paired"], complete["bootstrap_B"]),
                             ("complete_after_all_guards", 20, 13, 8))
            self.assertEqual(complete["provenance_sha256"],
                             c.sha(target / "provenance.json"))
            self.assertEqual(complete["output_manifest_sha256"],
                             c.sha(target / "output_manifest.json"))
            manifest = json.loads((target / "output_manifest.json").read_text())
            self.assertEqual(set(manifest["artifacts"]),
                             {p.name for p in target.iterdir() if p.is_file()} -
                             {"COMPLETE.json", "output_manifest.json"})
            for name, info in manifest["artifacts"].items():
                self.assertEqual(c.sha(target / name), info["sha256"])
                self.assertEqual((target / name).stat().st_size, info["bytes"])
            pairs = json.loads((target / "paired_differences.json").read_text())
            self.assertEqual(len(pairs), len(c.EXPECTED_PAIRS))
            self.assertEqual({(row["cohort"], row["population"],
                               row["alternative_configuration"]) for row in pairs},
                             c.EXPECTED_PAIRS)
            self.assertEqual(len({row["comparison_id"] for row in pairs}), len(pairs))
            for cohort, population, method in (("stv", "main_696", "whole_slide_ot"),
                                                ("kidrare", "amended_1581",
                                                 "whole_slide_ot")):
                report = json.loads((target / f"{cohort}_{population}_{method}.json").read_text())
                self.assertEqual(report["confidence_intervals_95"][
                    "macro_precision_at_5"]["B_effective"], 8)
                self.assertEqual(report["bootstrap"]["B"], 8)
                self.assertEqual(report["bootstrap"]["resampling_unit"],
                                 "supplied ST_ patient ID" if cohort == "stv"
                                 else "slide (nominal)")
        finally:
            if target.exists():
                shutil.rmtree(target)
        self.assertFalse(target.exists())
        self.assertEqual({path: snapshot(path) for path in (production, failed)}, before)


if __name__ == "__main__":
    unittest.main()
