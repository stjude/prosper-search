"""Focused R1.2 policy and arithmetic checks; never starts the full correction."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import r1_2_scope_correction as correction
from prosper.evaluation import external_cli, resample


class ScopeCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = correction.ROOT / "prosper_revision_output"
        cls.source = {name: correction.read_csv(out / path)
                      for name, path in correction.SOURCES.items()}
        cls.source["ccdi_metadata"] = correction.read_csv(
            correction.EXTERNAL["ccdi_metadata"])
        cls.policy = correction.read_csv(correction.POLICY)

    def test_approved_membership_and_source_counts(self):
        d = correction.build_decisions(self.source, self.policy)
        self.assertEqual(len(d), 765)
        self.assertEqual({col: int(d[col].sum()) for col in (
            "ffpe_scope_clear", "ffpe_add_site_only", "ffpe_add_all_unresolved",
            "allprep_scope_clear")}, {"ffpe_scope_clear": 524, "ffpe_add_site_only": 536,
                                      "ffpe_add_all_unresolved": 542,
                                      "allprep_scope_clear": 735})
        main = d[d.ffpe_scope_clear]
        self.assertEqual(main.tumor_group.value_counts()["NB"], 89)
        self.assertEqual(main.tumor_group.nunique(), 12)
        self.assertEqual(int(main.tumor_group.eq("NF").sum()), 0)
        self.assertNotIn("NF", main.tumor_group.unique())
        self.assertEqual(set(d.loc[d.scope_status.eq("UNRESOLVED"), "slide_id"]),
                         correction.EXPECTED_IDS - correction.EXPECTED_FOXR2)
        self.assertFalse(d.loc[d.slide_id.isin(correction.EXPECTED_FOXR2),
                               ["ffpe_scope_clear", "ffpe_add_site_only",
                                "ffpe_add_all_unresolved", "allprep_scope_clear"]].any().any())
        self.assertEqual(d.preparation_status.value_counts().to_dict(),
                         {"VERIFIED_FFPE": 546, "FROZEN_OCT": 209,
                          "MIXED": 6, "NONE_UNKNOWN": 4})
        self.assertEqual(d.loc[d.preparation_status.eq("NONE_UNKNOWN"),
                               "prep_manifest"].eq("Cytospin Slide").sum(), 3)
        self.assertTrue(d.loc[d.Diagnosis.eq("Ewing sarcoma") &
                              d["Diagnosis Category"].str.contains("CNS", case=False),
                              "scope_status"].eq("CLEAR").all())

    def test_duplicate_missing_and_source_mismatch_stop(self):
        bad = self.policy.copy()
        bad.loc[1, "slide_id"] = bad.loc[0, "slide_id"]
        with self.assertRaisesRegex(ValueError, "duplicate|ID set"):
            correction.build_decisions(self.source, bad)
        bad = self.policy.copy()
        bad.loc[0, "slide_id"] = "ABSENT"
        with self.assertRaisesRegex(ValueError, "ID set"):
            correction.build_decisions(self.source, bad)
        bad = self.policy.copy()
        bad.loc[0, "expected_site"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "expected_site"):
            correction.build_decisions(self.source, bad)
        mutated = dict(self.source)
        mutated["manifest_inscope"] = self.source["manifest_inscope"].copy()
        mutated["manifest_inscope"].loc[0, "tumor_group"] = "WRONG"
        with self.assertRaises(ValueError):
            correction.build_decisions(mutated, self.policy)

    def test_all_query_manifest_original_mapping_and_scope(self):
        d = correction.build_decisions(self.source, self.policy)
        all_rows = correction.build_all_manifest(self.source, d)
        self.assertEqual((len(all_rows), int(all_rows.in_original_inscope.sum())),
                         (4160, 765))
        original = self.source["manifest_all"].set_index("slide_id")
        self.assertTrue(all_rows.set_index("slide_id")[original.columns].equals(original))
        outside = all_rows[~all_rows.in_original_inscope]
        self.assertEqual(len(outside), 3395)
        self.assertTrue(outside.scope_status.eq(
            "NOT_ASSESSED_OUTSIDE_ORIGINAL_INSCOPE").all())
        self.assertFalse(outside[["legacy_765", "original_ffpe_546",
                                  "ffpe_scope_clear", "ffpe_add_site_only",
                                  "ffpe_add_all_unresolved", "allprep_scope_clear"]].any().any())
        fox = all_rows[all_rows.slide_id.isin(correction.EXPECTED_FOXR2)]
        self.assertEqual(len(fox), 6)
        self.assertTrue(fox.tumor_group.eq("NB").all())
        self.assertTrue(fox.legacy_765.all())
        self.assertFalse(fox.ffpe_scope_clear.any())
        mutated = dict(self.source)
        mutated["slide_preparation"] = self.source["slide_preparation"].iloc[1:]
        with self.assertRaisesRegex(ValueError, "coverage"):
            correction.build_all_manifest(mutated, d)

    def test_no_output_overwrite_and_root_requirement(self):
        with tempfile.TemporaryDirectory(dir=correction.ROOT) as scratch:
            dest = Path(scratch) / "new"
            correction.refuse_nonempty(dest)
            dest.mkdir()
            correction.refuse_nonempty(dest)
            (dest / "existing").write_text("source")
            with self.assertRaisesRegex(ValueError, "Refusing nonempty"):
                correction.refuse_nonempty(dest)
        old = os.environ.pop("PROSPER_OUT", None)
        try:
            with self.assertRaisesRegex(ValueError, "explicitly"):
                correction.output_root()
            os.environ["PROSPER_OUT"] = str(correction.ROOT.parent)
            with self.assertRaisesRegex(ValueError, "must resolve"):
                correction.output_root()
        finally:
            if old is not None:
                os.environ["PROSPER_OUT"] = old
            else:
                os.environ.pop("PROSPER_OUT", None)

    def test_old_nb182_membership_and_marginals(self):
        source = dict(self.source)
        for key in ("mapping_v02", "mapping_v03", "original_prediction"):
            source[key] = correction.read_csv(correction.EXTERNAL[key])
        d = correction.build_decisions(source, self.policy)
        with tempfile.TemporaryDirectory(dir=correction.ROOT) as scratch:
            dest = Path(scratch)
            correction.original_nb(source, dest, d)
            old = pd.read_csv(dest / "original_nb182_membership_and_source.csv")
            membership = pd.read_csv(dest / "nb_membership_reconciliation.csv")
            self.assertEqual((len(old), old.legacy_correct_fine_label_k10.sum()),
                             (182, 64))
            self.assertEqual(
                int(old.legacy_region_source_heuristic.eq("Brain / CNS").sum()), 14)
            self.assertEqual(membership.old_nb182.sum(), 182)
            self.assertEqual(membership.current_nb178.sum(), 178)
            self.assertEqual(membership.main_nb89.sum(), 89)
            original = source["original_prediction"].copy()
            original.loc[1, "Scan ID"] = original.loc[0, "Scan ID"]
            bad = dict(source, original_prediction=original)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                correction.original_nb(bad, dest, d)
            original = source["original_prediction"].copy()
            original.loc[0, "Scan ID"] = "MISSINGID_0"
            bad = dict(source, original_prediction=original)
            with self.assertRaisesRegex(ValueError, "missing participant"):
                correction.original_nb(bad, dest, d)

    def test_strict_numeric_fidelity_tolerance(self):
        actual = pd.DataFrame({"query": ["q"], "distance": [0.7300000000000]})
        correction.compare_frames(actual, actual.copy(), "distance")
        changed = actual.copy()
        changed.loc[0, "distance"] += 2e-12
        with self.assertRaisesRegex(ValueError, "distance"):
            correction.compare_frames(actual, changed, "distance")
        changed.loc[0, "distance"] = actual.loc[0, "distance"] + 4e-13
        correction.compare_frames(actual, changed, "distance")

    def test_input_hash_snapshot_rejects_midrun_changes(self):
        with tempfile.TemporaryDirectory(dir=correction.ROOT) as scratch:
            path = Path(scratch) / "fixture.csv"
            path.write_text("original\n")
            paths = {"fixture": path}
            initial = {"fixture": correction.sha256(path)}
            correction.verify_input_hashes(paths, initial)
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "changed during execution"):
                correction.verify_input_hashes(paths, initial)

    def test_archived_ffpe546_ot_point_summary(self):
        d = correction.build_decisions(self.source, self.policy)
        q = pd.read_csv(correction.ROOT / "prosper_revision_output/external_ccdi_inscope/"
                        "per_query_ccdi_uni2_ot_cosine.csv")
        archival = correction.verify_ffpe546_ot(d, q)
        self.assertEqual((archival["n"], archival["observed_classes"]), (546, 13))
        bad = q.copy()
        idx = bad.index[bad["query"].eq(d.loc[d.original_ffpe_546, "slide_id"].iloc[0])][0]
        bad.loc[idx, "hit_k5"] = 1 - bad.loc[idx, "hit_k5"]
        with self.assertRaisesRegex(ValueError, "Archived C3"):
            correction.verify_ffpe546_ot(d, bad)

    def test_nb_descriptive_missingness_keeps_raw_source(self):
        d = correction.build_decisions(self.source, self.policy)
        q = pd.read_csv(correction.ROOT / "prosper_revision_output/external_ccdi_inscope/"
                        "per_query_ccdi_uni2_ot_cosine.csv")
        with tempfile.TemporaryDirectory(dir=correction.ROOT) as scratch:
            correction.nb_descriptives(Path(scratch), d, q)
            tab = correction.read_csv(Path(scratch) / "main_nb_descriptive_factors.csv")
            missing = tab[(tab.factor == "Treatment Type") &
                          (tab.source_value == "") &
                          (tab.source_value_missing == "True")]
            self.assertEqual(int(missing.n.sum()), 87)
            self.assertEqual(tab[tab.factor == "Diagnosis"].n.astype(int).sum(), 89)
            confusion = correction.read_csv(
                Path(scratch) / "main_nb_raw_diagnosis_closest_vote5_confusion.csv")
            self.assertIn("source_diagnosis_missing", confusion)
            self.assertIn("pred_closest_k5_missing", confusion)

    @staticmethod
    def fixture():
        pop = pd.DataFrame({"slide_id": ["q1", "q2", "q3", "q4"],
                            "participant": ["p1", "p2", "p3", "p4"],
                            "tumor_group": ["A", "A", "B", "B"]})
        ranks = np.array([
            ["A"] * 10, ["B", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            ["A"] * 10, ["B"] * 10], dtype=object)
        q = pd.DataFrame({"query": pop.slide_id, "true_label": pop.tumor_group,
                          "pred_generous_k5": ["A", "A", "A", "B"],
                          "pred_absolute_k5": ["A", "A", "A", "B"],
                          "pred_closest_k5": ["A", "A", "A", "B"]})
        q.attrs["ranked_labels"] = ranks
        cfg = {"library_classes": ["A", "B", "NF"],
               "library_class_counts": {"A": 20, "B": 20, "NF": 20}}
        return pop, q, cfg

    def test_bootstrap_vectorization_and_paired_same_draws(self):
        pop, q, cfg = self.fixture()
        b, seed = 23, 123
        weights = correction.draw_weights(pop, b, seed)
        result, draws = correction.score(pop, q, cfg, weights)
        self.assertEqual([r["tumor_group"] for r in
                          result["per_class_hit_precision"]], ["A", "B"])
        self.assertEqual(result["voting_k5"]["closest"]["macro_f1"],
                         f1_score(pop.tumor_group, q.pred_closest_k5,
                                  labels=cfg["library_classes"], average="macro",
                                  zero_division=0))
        index = resample.build_patient_index(pop, "slide_id", "tumor_group", "participant")
        rng = np.random.default_rng(seed)
        direct_macro = []
        direct_micro = []
        direct_f1 = []
        direct_perclass_a = []
        truth = pop.tumor_group.to_numpy()
        relevance = q.attrs["ranked_labels"] == truth[:, None]
        per_query = relevance[:, :5].any(axis=1).astype(float)
        for i in range(b):
            sample = resample.sample_queries(index, ["A", "B"], rng,
                                             mode="pps_cluster_one_slide")
            rows = np.array([pop.index[pop.slide_id.eq(x)][0] for x in sample])
            np.testing.assert_array_equal(weights[i], np.bincount(rows, minlength=4))
            direct_macro.append(np.mean([per_query[rows][truth[rows] == x].mean()
                                         for x in ("A", "B")]))
            direct_micro.append(per_query[rows].mean())
            direct_perclass_a.append(per_query[rows][truth[rows] == "A"].mean())
            direct_f1.append(f1_score(truth[rows], q.pred_closest_k5.to_numpy()[rows],
                                      average="macro", labels=cfg["library_classes"],
                                      zero_division=0))
        np.testing.assert_allclose(draws["macro_hit_at_5"], direct_macro)
        np.testing.assert_allclose(draws["micro_hit_at_5"], direct_micro)
        np.testing.assert_allclose(draws["closest_macro_f1_k5"], direct_f1)
        self.assertEqual(result["confidence_intervals_95"]["per_class_A_hit_at_5"],
                         correction.interval(np.array(direct_perclass_a)))
        changed = q.copy()
        changed.attrs["ranked_labels"] = q.attrs["ranked_labels"].copy()
        changed.attrs["ranked_labels"][2] = ["B"] * 10
        changed.loc[2, ["pred_closest_k5", "pred_absolute_k5",
                        "pred_generous_k5"]] = "B"
        different, other = correction.score(pop, changed, cfg, weights)
        actual_pair = correction.paired(
            {"point": {"micro_hit_at_5": different["point_estimates"]["micro_hit_at_5"],
                       "macro_hit_at_5": different["point_estimates"]["macro_hit_at_5"],
                       "micro_precision_at_5": different["point_estimates"]["micro_precision_at_5"],
                       "macro_precision_at_5": different["point_estimates"]["macro_precision_at_5"],
                       "generous_accuracy_k5": different["voting_k5"]["generous"]["accuracy"],
                       "macro_generous_accuracy_k5": different["macro_generous_accuracy_k5"]},
             "draws": other},
            {"point": {"micro_hit_at_5": result["point_estimates"]["micro_hit_at_5"],
                       "macro_hit_at_5": result["point_estimates"]["macro_hit_at_5"],
                       "micro_precision_at_5": result["point_estimates"]["micro_precision_at_5"],
                       "macro_precision_at_5": result["point_estimates"]["macro_precision_at_5"],
                       "generous_accuracy_k5": result["voting_k5"]["generous"]["accuracy"],
                       "macro_generous_accuracy_k5": result["macro_generous_accuracy_k5"]},
             "draws": draws})
        # One B query changes from a miss to a hit; class B has two draws.
        direct_delta = weights[:, 2].astype(float) / 4
        np.testing.assert_allclose(other["micro_hit_at_5"] - draws["micro_hit_at_5"],
                                   direct_delta)
        self.assertEqual(actual_pair["micro_hit_at_5"]["ci_95"],
                         correction.interval(direct_delta))
        self.assertEqual(actual_pair["macro_hit_at_5"]["ci_95"],
                         correction.interval(weights[:, 2].astype(float) / 4))

    def test_tied_vote_and_abstention_macro_micro_f1_same_draws(self):
        pop, q, cfg = self.fixture()
        cfg["library_classes"] = ["A", "B", "C", "NF"]
        q.attrs["ranked_labels"] = np.array([
            ["A", "B", "A", "B", "C"] + ["C"] * 5,
            ["A"] * 10,
            ["A", "B", "B", "A", "C"] + ["C"] * 5,
            ["B"] * 10], dtype=object)
        for ix, truth in enumerate(pop.tumor_group):
            top = q.attrs["ranked_labels"][ix, :5].tolist()
            closest, tied, *_ = external_cli.vote(top)
            q.loc[ix, "pred_closest_k5"] = closest
            q.loc[ix, "pred_absolute_k5"] = external_cli.strict_majority(top, 5)
            q.loc[ix, "pred_generous_k5"] = truth if truth in tied else closest
        self.assertEqual(q.loc[0, "pred_absolute_k5"], "Ambiguous")
        self.assertEqual(q.loc[0, "pred_closest_k5"], "A")
        self.assertEqual(q.loc[2, "pred_generous_k5"], "B")
        weights = correction.draw_weights(pop, 29, 43)
        report, draws = correction.score(pop, q, cfg, weights)
        self.assertNotIn("NF", [row["tumor_group"] for row in
                                 report["per_class_hit_precision"]])
        self.assertEqual(len(report["voting_k5"]["absolute"]["per_class_f1"]), 4)
        index = resample.build_patient_index(pop, "slide_id", "tumor_group", "participant")
        rng = np.random.default_rng(43)
        for i in range(len(weights)):
            samples = resample.sample_queries(index, ["A", "B"], rng,
                                              mode="pps_cluster_one_slide")
            rows = np.array([pop.index[pop.slide_id.eq(x)][0] for x in samples])
            for rule in ("generous", "absolute", "closest"):
                pred = q[f"pred_{rule}_k5"].to_numpy()[rows]
                for avg in ("micro", "macro"):
                    direct = f1_score(pop.tumor_group.to_numpy()[rows], pred,
                                      labels=cfg["library_classes"], average=avg,
                                      zero_division=0)
                    self.assertAlmostEqual(draws[f"{rule}_{avg}_f1_k5"][i], direct)

    def test_historical_score_has_no_bootstrap_or_nf_fabrication(self):
        pop, q, cfg = self.fixture()
        report, draws = correction.score(pop, q, cfg, None)
        self.assertEqual(report["confidence_intervals_95"], {})
        self.assertEqual(draws, {})
        self.assertNotIn("NF", [r["tumor_group"] for r in
                                 report["per_class_hit_precision"]])

    def test_common_twelve_class_context_all_six_populations(self):
        d = correction.build_decisions(self.source, self.policy)
        base = correction.ROOT / "prosper_revision_output/external_ccdi_inscope"
        r = pd.read_csv(base / "ranked_lists_ccdi_uni2_ot_cosine.csv.gz",
                        dtype={"query": str, "neighbour": str})
        cfg = json.loads((base / "external_ccdi_uni2_ot_cosine.json").read_text())["config"]
        populations = ("legacy_765", "original_ffpe_546", "ffpe_scope_clear",
                       "ffpe_add_site_only", "ffpe_add_all_unresolved",
                       "allprep_scope_clear")
        context = correction.common_class_context(
            d, {"whole_slide_ot": r.set_index(["query", "rank"])},
            {"whole_slide_ot": cfg}, populations)
        self.assertEqual(context.population.tolist(), list(populations))
        self.assertEqual(context.n_common_class.tolist(), [762, 543, 524, 534, 539, 735])
        self.assertEqual(context.n_excluded_noncommon.tolist(), [3, 3, 0, 2, 3, 0])
        self.assertTrue(context.descriptive_only.all())
        self.assertEqual(context.n_queries.tolist(), context.n_common_class.tolist())
        self.assertEqual(context.common_classes.str.split("|", regex=False).map(len).tolist(),
                         [12] * 6)
        self.assertIn("micro_precision_at_10", context.columns)
        self.assertIn("macro_mrr", context.columns)
        self.assertEqual(context.retrieval_depth.tolist(), [10] * 6)


if __name__ == "__main__":
    unittest.main()
