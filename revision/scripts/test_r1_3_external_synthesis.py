"""Isolated deterministic fixtures for the guarded [decision record] synthesis."""

import json
import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import r1_3_external_synthesis as s


def population(labels, patients=None, cohort="stv"):
    ids = [f"q{i}" for i in range(len(labels))]
    return pd.DataFrame({"slide_id": ids, "tumor_group": labels,
                         "patient_id": patients or ids,
                         "participant": ids,
                         "preparation_status": "VERIFIED_FFPE",
                         "pathpt_subtype": "",
                         "pathpt_only_new_label": "False"})


def analysis(labels, top5=None, patients=None, cohort="stv",
             configuration="whole_slide_ot", empty=(), role="main",
             population_name="tiny"):
    pop = population(labels, patients, cohort)
    ranks = []
    for sid, true in zip(pop.slide_id, labels):
        if sid in empty:
            continue
        first = top5.get(sid, [true] * 5) if top5 else [true] * 5
        for i, label in enumerate(first + ["OTHER"] * (10 - len(first)), 1):
            ranks.append({"query": sid, "rank": i, "neighbour": f"{sid}_ref{i}",
                          "neighbour_label": label, "distance": float(i)})
    r = pd.DataFrame(ranks, columns=["query", "rank", "neighbour",
                                     "neighbour_label", "distance"])
    ids = [q for q in pop.slide_id if q not in empty]
    lookup = dict(zip(pop.slide_id, labels))
    q = s.external_cli.per_query_frame(
        ids, [lookup[q] for q in ids],
        r.neighbour.to_numpy().reshape(len(ids), 10),
        r.neighbour_label.to_numpy().reshape(len(ids), 10),
        r.distance.to_numpy().reshape(len(ids), 10), (1, 3, 5, 10), 10)
    return s.Analysis(cohort, population_name, configuration, role,
                      pop, r, q, set(empty), "fixture")


class SynthesisFixtures(unittest.TestCase):
    def test_vote_tie_and_rank_endpoints(self):
        a = analysis(["WT"], {"q0": ["NB", "WT", "NB", "WT", "NORMAL"]})
        s.validate_ranking(a.ranked, a.query, {"q0": "WT"}, "fixture")
        inv = s.inventory(a, a.target)
        row = inv.iloc[0]
        self.assertEqual(row.relevant_top5_count, 2)
        self.assertEqual(row.precision_at_5, .4)
        self.assertEqual(row.hit_at_5, 1)
        self.assertEqual(row.first_relevant_rank, "2-5")
        self.assertEqual(row.closest_prediction, "NB")
        self.assertFalse(row.closest_correct)
        self.assertTrue(row.generous_correct)
        self.assertIn("WT", row.tied_set_k5)

    def test_empty_vs_unobserved_and_technical(self):
        a = analysis(["WT", "NB"], {"q0": ["NB"] * 5}, empty=("q1",))
        rows = s.inventory(a, a.target).set_index("query")
        self.assertEqual(rows.at["q0", "first_relevant_rank"],
                         "not_observed_through_10")
        self.assertIsNone(rows.at["q1", "first_relevant_rank"])
        self.assertTrue(pd.isna(rows.at["q1", "precision_at_5"]))
        self.assertFalse(rows.at["q1", "technical_failure"])
        self.assertTrue(rows.at["q1", "confirmed_empty"])
        a.empty.clear()
        with self.assertRaisesRegex(ValueError, "technical failures"):
            s.inventory(a, a.target)

    def test_first_relevant_rank_six_to_ten(self):
        a = analysis(["WT"], {"q0": ["NB"] * 6 + ["WT"] + ["NB"] * 3})
        row = s.inventory(a, a.target).iloc[0]
        self.assertEqual(row.first_relevant_rank, "6-10")
        self.assertEqual(row.hit_at_5, 0)
        self.assertEqual(row.relevant_top5_count, 0)
        self.assertEqual(row.beyond_top10_cutoff_tie, "unavailable_not_saved")

    def test_conditional_composition_and_failure_denominators(self):
        a = analysis(["WT", "WT"],
                     {"q0": ["WT", "NB", "NB", "NB", "NB"],
                      "q1": ["NB"] * 5})
        inv = s.inventory(a, a.target)
        views = s.failure_views(a, inv)
        matching = next(x for x in views["composition"]
                        if x["neighbour_label"] == "WT")
        self.assertEqual((matching["slot_count"],
                          matching["denominator_top5_slots"]), (1, 10))
        fail = next(x for x in views["failure_composition"]
                    if x["endpoint"] == "hit_miss" and
                    x["neighbour_label"] == "NB")
        self.assertEqual((fail["slot_count"], fail["n_failure_queries"],
                          fail["denominator_failure_top5_slots"]), (5, 1, 5))
        exposure = next(x for x in views["recurrence"]
                        if x["reference_id"] == "q0_ref1" and
                        x["endpoint"] == "hit_miss")
        self.assertEqual((exposure["all_query_exposure"],
                          exposure["failure_exposure"]), (1, 0))

    def test_endpoint_specific_transitions_and_separate_loss(self):
        before = analysis(["WT", "WT", "WT"],
                          {f"q{i}": ["NB"] * 5 for i in range(3)})
        after = analysis(["WT", "WT", "WT"], {
            "q0": ["WT"] * 5, "q1": ["NB"] * 5}, empty=("q2",),
            configuration="query_only")
        rows, losses = s.transitions(after, s.inventory(after, after.target),
                                    s.inventory(before, before.target))
        self.assertEqual(len(losses), 1)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(x["transition"] == "fixed" for x in rows), 3)
        self.assertNotIn("q2", {x["query"] for x in rows})

    def test_cluster_multiplicity_and_variable_denominator(self):
        pop = population(["WT", "WT", "WT", "NB", "NB"],
                         ["P1", "P1", "P2", "P3", "P4"])
        w, info = s.weights(pop, "stv", s.Spec(b=40))
        self.assertTrue(np.array_equal(w[:, 0], w[:, 1]))
        self.assertGreater(info["max_resampled_slides"],
                           info["min_resampled_slides"])
        result = s.ratio(np.array([1., 1., 0., 0., 1.]), pop, w, ["NB", "WT"])
        self.assertTrue(np.allclose(result["micro"][1],
                                    (w @ np.array([1., 1., 0., 0., 1.])) /
                                    w.sum(axis=1)))
        nominal, text = s.weights(pop, "kidrare", s.Spec(b=40))
        self.assertTrue(np.any(nominal[:, 0] != nominal[:, 1]))
        self.assertIn("nominal slide", text["unit"])

    def test_ccdi_accepted_draw_order_and_shared_population(self):
        pop = population(["WT", "WT", "NB"], cohort="ccdi")
        spec = s.Spec(b=12)
        cache = {}
        w, desc = s.shared_weights(cache, pop, "ccdi", spec)
        self.assertTrue(np.array_equal(w, s.ccdi.draw_weights(pop, b=12, seed=123)))
        self.assertIs(w, s.shared_weights(cache, pop, "ccdi", spec)[0])
        full_target, _ = s.shared_weights(cache, pop, "ccdi", spec, "full_target")
        self.assertIsNot(w, full_target)
        self.assertTrue(np.array_equal(w, full_target))
        self.assertIn("singleton", desc["unit"])
        pop.loc[0, "participant"] = pop.loc[1, "participant"]
        with self.assertRaisesRegex(ValueError, "truth changed|participant"):
            s.weights(pop, "ccdi", spec)

    def test_cached_weights_validate_units_seed_and_draw_count(self):
        pop = population(["WT"] * 10, ["P1"] * 5 + ["P2"] * 5)
        cache = {}
        first, _ = s.shared_weights(cache, pop, "stv", s.Spec(b=10))
        small = s.estimate_rows(pop, first, np.ones(10), "hit_at_5",
                                {"cohort": "stv"}, ["WT"])
        self.assertEqual((small[0]["n_slides"], small[0]["n_resampling_units"]),
                         (10, 2))
        self.assertTrue(small[0]["sparse_units"])
        self.assertTrue(small[0]["degenerate_interval"])
        remapped = pop.copy()
        remapped.loc[0, "patient_id"] = "P2"
        self.assertIsNot(first, s.shared_weights(cache, remapped, "stv",
                                                s.Spec(b=10))[0])
        self.assertIsNot(first, s.shared_weights(cache, pop, "stv",
                                                s.Spec(b=11))[0])
        self.assertIsNot(first, s.shared_weights(cache, pop, "stv",
                                                s.Spec(b=10, seed=124))[0])
        mixed = pop.copy()
        mixed.loc[0, "patient_id"] = ""
        with self.assertRaisesRegex(ValueError, "resampling unit"):
            s.shared_weights(cache, mixed, "stv", s.Spec(b=10))
        ccdi = population(["WT", "WT"], cohort="ccdi")
        s.shared_weights(cache, ccdi, "ccdi", s.Spec(b=10))
        ccdi.loc[0, "participant"] = ccdi.loc[1, "participant"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            s.shared_weights(cache, ccdi, "ccdi", s.Spec(b=10))

    def test_class_scoped_coverage_counts(self):
        base = analysis(["LIVER", "LIVER", "WT"])
        alt = analysis(["LIVER", "LIVER", "WT"], empty={"q0"},
                       configuration="query_only")
        _, rows = s.summarize(alt, s.inventory(alt, alt.target), s.Spec(b=16),
                              {}, s.inventory(base, base.target))
        table = {r["tumor_group"]: r for r in rows if r["metric"] == "coverage"}
        self.assertEqual([(table[g]["n_slides"], table[g]["n_returned"],
                           table[g]["n_confirmed_empty"]) for g in (
                           "LIVER", "WT", "micro", "macro")],
                         [(2, 1, 1), (1, 1, 0), (3, 2, 1), (3, 2, 1)])

    def test_full_target_coverage_and_conditional_pair_separate_draws(self):
        base = analysis(["WT", "WT", "NB", "NB"])
        alternative = analysis(["WT", "WT", "NB", "NB"],
                               empty=("q0",), configuration="query_only")
        base_inv = s.inventory(base, base.target)
        inv = s.inventory(alternative, alternative.target)
        primary, cov = s.summarize(alternative, inv, s.Spec(b=30), {},
                                   base_inv)
        micro = [r for r in cov if r["metric"] == "coverage" and
                 r["tumor_group"] == "micro"][0]
        self.assertEqual(micro["point"], .75)
        self.assertEqual(micro["baseline_point"], 1)
        conditional = [r for r in primary if r["metric"] == "precision_at_5" and
                       r["tumor_group"] == "micro"][0]
        self.assertEqual(conditional["n"], 3)
        self.assertNotEqual(conditional["weights_sha256"], micro["weights_sha256"])
        self.assertEqual(conditional["alternative_configuration"], "query_only")
        self.assertEqual(conditional["baseline_configuration"], "whole_slide_ot")
        self.assertEqual(conditional["comparison_id"],
                         "stv:tiny:query_only:vs:whole_slide_ot")

    def test_paired_replicates_not_ci_subtraction(self):
        pop = population(["WT", "WT", "NB", "NB"])
        w, _ = s.weights(pop, "kidrare", s.Spec(b=40))
        v = np.array([1., 0., 1., 0.])
        r = s.estimate_rows(pop, w, v, "hit_at_5",
                            {"cohort": "kidrare"}, ["NB", "WT"], v)
        self.assertTrue(all(x["difference_ci_lo"] == x["difference_ci_hi"] == 0
                            for x in r))
        self.assertTrue(any(x["ci_hi"] > x["ci_lo"] for x in r))
        self.assertEqual({x["tumor_group"] for x in r},
                         {"NB", "WT", "micro", "macro"})

    def test_guards_missing_cross_class_duplicate(self):
        p = population(["WT", "NB"])
        p.loc[0, "patient_id"] = ""
        with self.assertRaises(ValueError):
            s.weights(p, "stv", s.Spec(b=4))
        p.loc[0, "patient_id"] = "same"
        p.loc[1, "patient_id"] = "same"
        with self.assertRaisesRegex(ValueError, "Cross-class"):
            s.weights(p, "stv", s.Spec(b=4))
        p.loc[1, "slide_id"] = p.loc[0, "slide_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            s.weights(p, "kidrare", s.Spec(b=4))

    def test_fixed_class_and_undefined_draw(self):
        p = population(["WT", "NB"])
        with self.assertRaisesRegex(ValueError, "Missing fixed class"):
            s.ratio(np.array([1., 0.]), p, np.array([[1, 0]]), ["NB", "WT"])
        with self.assertRaisesRegex(ValueError, "classes mismatch"):
            s.ratio(np.array([1., 0.]), p, np.array([[1, 1]]), ["WT"])

    def test_full_gate_review_missing_and_tampered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / s.full.DEST).mkdir()
            with self.assertRaises(ValueError):
                s.full_gate(root, s.CONTRACT)

    def test_normal_targets_and_new_wt_not_legacy_flags(self):
        a = analysis(["LIVER", "NORMAL", "WT"], cohort="kidrare")
        both = a.target[a.target.tumor_group.ne("NORMAL")]
        self.assertEqual(set(a.target.slide_id) - set(both.slide_id), {"q1"})
        self.assertEqual(s.Spec().kid_main - s.Spec().kid_both, 50)
        self.assertEqual(s.Spec().new_kid, 354)

    def test_preparation_strata_points_no_intervals(self):
        a = analysis(["WT", "NB", "WT", "NB"], cohort="ccdi",
                     population_name="allprep_scope_clear",
                     role="preparation_sensitivity")
        a.target["preparation_status"] = list(s.PREPARATIONS)
        inv = s.inventory(a, a.target)
        rows, _ = s.descriptive(a, inv, set(), s.Spec(b=10))
        self.assertEqual({x["stratum"] for x in rows}, set(s.PREPARATIONS))
        self.assertTrue(all(x["interval"] is None and x["test"] is None
                            for x in rows))

    def test_five_wt_patterns_include_new_false_legacy_flag(self):
        a = analysis(["WT"] * 5 + ["LIVER"],
                     {"q5": ["WT"] * 5}, cohort="kidrare",
                     population_name="amended_1581")
        a.target["pathpt_subtype"] = list(s.PATTERNS) + [""]
        a.target["pathpt_only_new_label"] = ["True"] + ["False"] * 5
        a.target["query_only_eligible"] = "False"
        a.target["source_group"] = (["ADDED_PATHPT"] + ["ORIGINAL_KEEP"] * 4 +
                                    ["NOT_APPLICABLE_NON_WT"])
        inv = s.inventory(a, a.target)
        rows, context = s.descriptive(a, inv, {"q0"}, s.Spec(b=8))
        keep = next(r for r in rows if r["stratum_type"] == "source_group"
                    and r["stratum"] == "ORIGINAL_KEEP" and
                    r["endpoint"] == "hit_yield")
        self.assertEqual((keep["n_target"], keep["point"]), (4, 1.))
        self.assertEqual(inv.loc[inv.tumor_group.eq("LIVER"),
                                 "source_group"].tolist(),
                         ["NOT_APPLICABLE_NON_WT"])
        self.assertEqual(len(context), 10)
        self.assertEqual(sum(row["n_queries"] for row in context), 5)
        self.assertEqual(sum(row["n_queries"] for row in context
                             if row["source_group"] == "ADDED_PATHPT"), 1)
        self.assertTrue(all(row["interval"] is None for row in rows))
        self.assertEqual({row["stratum"] for row in rows
                          if row["stratum_type"] == "wt_pathpt_pattern"},
                         set(s.PATTERNS))

    def test_duplicate_rank_missing_reference_and_label_conflict(self):
        a = analysis(["WT"])
        bad = a.ranked.copy()
        bad.loc[0, "neighbour"] = bad.loc[1, "neighbour"]
        with self.assertRaisesRegex(ValueError, "duplicate ranking"):
            s.validate_ranking(bad, a.query, {"q0": "WT"}, "fixture")
        with self.assertRaisesRegex(ValueError, "truth/query"):
            s.validate_ranking(a.ranked, a.query, {"q0": "NB"}, "fixture")
        reference = dict(zip(a.ranked.neighbour, a.ranked.neighbour_label))
        reference[a.ranked.neighbour.iloc[0]] = "NB"
        with self.assertRaisesRegex(ValueError, "reference"):
            s.validate_ranking(a.ranked, a.query, {"q0": "WT"},
                               "fixture", reference)

    def test_saved_report_and_rank_consistency_and_tampering(self):
        a = analysis(["WT", "NB"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "fixture"
            qpath = root / f"{prefix}_per_query.csv"
            rankpath = root / f"{prefix}_ranked.csv.gz"
            reportpath = root / f"{prefix}.json"
            a.query.to_csv(qpath, index=False)
            a.ranked.to_csv(rankpath, index=False, compression={
                "method": "gzip", "mtime": 0})
            report = {
                "point_estimates": {"n_queries": 2, "micro_hit_at_5": 1.,
                                    "micro_precision_at_5": 1.,
                                    "macro_hit_at_5": 1., "macro_precision_at_5": 1.},
                "per_class_primary": [
                    {"tumor_group": name, "hit_at_5": 1., "precision_at_5": 1.}
                    for name in ("WT", "NB")]}
            reportpath.write_text(json.dumps(report))
            manifest = {"artifacts": {
                p.name: {"sha256": s.digest(p), "bytes": p.stat().st_size}
                for p in (qpath, rankpath, reportpath)}}
            truth = dict(zip(a.target.slide_id, a.target.tumor_group))
            s.accepted_ranking(root, prefix, truth, set(truth), {}, manifest)
            report["per_class_primary"][0]["precision_at_5"] = .5
            reportpath.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                s.accepted_ranking(root, prefix, truth, set(truth), {}, manifest)
            with self.assertRaisesRegex(ValueError, "per-group point"):
                s.accepted_ranking(root, prefix, truth, set(truth), {})
            report["per_class_primary"][0]["precision_at_5"] = 1.
            reportpath.write_text(json.dumps(report))
            tampered = a.ranked.copy()
            tampered.loc[0, "neighbour_label"] = "NB"
            tampered.to_csv(rankpath, index=False, compression={
                "method": "gzip", "mtime": 0})
            with self.assertRaisesRegex(ValueError, "endpoint differs|point differs"):
                s.accepted_ranking(root, prefix, truth, set(truth), {})
            a.ranked.to_csv(rankpath, index=False, compression={
                "method": "gzip", "mtime": 0})
            original_read = s.read
            def changed_mid_read(path):
                result = original_read(path)
                if path == reportpath:
                    reportpath.write_text(json.dumps(
                        {**result, "per_class_primary": [
                            {**result["per_class_primary"][0], "n": 3},
                            result["per_class_primary"][1]]}))
                return result
            with patch.object(s, "read", side_effect=changed_mid_read):
                with self.assertRaisesRegex(ValueError, "changed during read"):
                    s.accepted_ranking(root, prefix, truth, set(truth), {})

    def test_frozen_helper_source_pin_rejects_bytes_and_foreign_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "helper.py"
            source.write_text("approved")
            record = {"path": str(source), "sha256": s.digest(source)}
            b = s.BoundInputs()
            s.bind_provenance_pin("ccdi_helper", record, b, source)
            source.write_text("altered")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                s.bind_provenance_pin("ccdi_helper", record, {}, source)
            source.write_text("approved")
            foreign = Path(tmp) / "foreign.py"
            foreign.write_text("approved")
            with self.assertRaisesRegex(ValueError, "path changed"):
                s.bind_provenance_pin("ccdi_helper", {
                    "path": str(foreign), "sha256": s.digest(foreign)}, {}, source)
            with self.assertRaisesRegex(ValueError, "Consumed input changed"):
                source.write_text("changed again")
                b.snapshot()

    def test_accepted_pin_cannot_bless_post_digest_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "helper.py"
            path.write_text("accepted")
            real_digest = s.digest
            record = {"path": str(path), "sha256": real_digest(path)}
            for name in ("fixture_helper", "ccdi_provenance",
                         "correction_provenance"):
                path.write_text("accepted")
                first = [True]
                def swap_after_hash(candidate):
                    value = real_digest(candidate)
                    if candidate == path and first[0]:
                        first[0] = False
                        path.write_text("changed immediately after approved hash")
                    return value
                b = s.BoundInputs()
                with patch.object(s, "digest", side_effect=swap_after_hash):
                    with self.assertRaisesRegex(ValueError, "bytes changed"):
                        s.bind_provenance_pin(name, record, b, path)
                self.assertNotEqual(
                    b.initial.get(f"pinned_{name}"), real_digest(path))

    def test_ccdi_pinned_rank_and_interval_semantic_replay(self):
        a = analysis(["WT", "NB"], cohort="ccdi",
                     configuration="query_tumor_ot",
                     population_name="ffpe_scope_clear")
        config = {"library_classes": ["NB", "OTHER", "WT"],
                  "library_class_counts": {"NB": 10, "OTHER": 10, "WT": 10}}
        pop = a.target
        weights = s.ccdi.draw_weights(pop, b=8, seed=123)
        q = a.query.copy()
        q.attrs["ranked_labels"] = a.ranked.neighbour_label.to_numpy().reshape(2, 10)
        report, _ = s.ccdi.score(pop, q, config, weights)
        report.update({"population": "ffpe_scope_clear",
                       "configuration": "query_tumor_ot", "n_participants": 2,
                       "observed_classes": ["NB", "WT"]})
        prefix = "query_tumor_ot_ffpe_scope_clear"
        source_labels = dict(zip(a.ranked.neighbour, a.ranked.neighbour_label))
        truth = dict(zip(pop.slide_id, pop.tumor_group))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a.query.to_csv(root / f"{prefix}_per_query.csv", index=False)
            a.ranked.to_csv(root / f"{prefix}_ranked.csv.gz", index=False,
                            compression={"method": "gzip", "mtime": 0})
            accepted = {prefix: {"report": copy.deepcopy(report)}}
            def check(reference=source_labels, ranked=a.ranked, target=pop):
                return s.verify_ccdi_configuration(
                    root, "query_tumor_ot", {"ffpe_scope_clear": target},
                    a.query, ranked, config, accepted,
                    {"ffpe_scope_clear": weights}, reference, truth)
            check()
            accepted[prefix]["report"]["confidence_intervals_95"][
                "micro_hit_at_5"]["lo"] = 0
            with self.assertRaisesRegex(ValueError, "accepted numerical result"):
                check()
            accepted[prefix]["report"] = copy.deepcopy(report)
            accepted[prefix]["report"]["point_estimates"]["micro_hit_at_5"] = 0
            with self.assertRaisesRegex(ValueError, "accepted numerical result"):
                check()
            accepted[prefix]["report"] = copy.deepcopy(report)
            wrong = dict(source_labels)
            wrong[a.ranked.neighbour.iloc[0]] = "NB"
            with self.assertRaisesRegex(ValueError, "reference"):
                check(reference=wrong)
            wrong_rank = a.ranked.copy()
            wrong_rank.loc[0, "neighbour"] = "foreign"
            with self.assertRaisesRegex(ValueError, "endpoint differs|reference"):
                check(ranked=wrong_rank)
            with self.assertRaisesRegex(ValueError, "pinned original query export"):
                check(target=pop.iloc[:1])

    def test_ccdi_paired_replay_rejects_ci_value_identity_and_coverage(self):
        pop = population(["WT", "NB"], cohort="ccdi")
        names = ("whole_slide_ot", "query_tumor_ot")
        keys = ("micro_hit_at_5", "macro_hit_at_5",
                "micro_precision_at_5", "macro_precision_at_5",
                "generous_accuracy_k5", "macro_generous_accuracy_k5")
        entries = {"tiny": {
            arm: {"point": {key: point for key in keys},
                  "draws": {key: np.array([point - .1, point, point + .1])
                            for key in keys}}
            for arm, point in zip(names, (.5, .75))}}
        configs = {arm: {"library_classes": ["NB", "WT"]} for arm in names}
        original = {
            "population": "tiny", "configuration": "query_tumor_ot",
            "baseline": "whole_slide_ot", "n_paired": 2,
            "difference": s.ccdi.paired(entries["tiny"]["query_tumor_ot"],
                                        entries["tiny"]["whole_slide_ot"]),
            "observed_truth_classes": ["NB", "WT"],
            "library_label_sets_equal": True,
            "note": "No macro F1 compared across distinct library label sets"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paired_differences.json"
            def check(rows):
                path.write_text(json.dumps(rows))
                s.verify_ccdi_paired(path, entries, {"tiny": pop}, configs, names)
            check([original])
            for field in ("difference", "ci", "population", "configuration",
                          "n_paired", "class_set"):
                row = copy.deepcopy(original)
                if field == "difference":
                    row["difference"]["micro_hit_at_5"]["difference"] += .1
                elif field == "ci":
                    row["difference"]["micro_hit_at_5"]["ci_95"]["lo"] += .1
                elif field == "population":
                    row["population"] = "other"
                elif field == "configuration":
                    row["configuration"] = "both_tumor_ot"
                elif field == "n_paired":
                    row["n_paired"] = 1
                else:
                    row["library_label_sets_equal"] = False
                with self.subTest(field=field), self.assertRaises(ValueError):
                    check([row])
            for rows in ([], [original, original]):
                with self.assertRaisesRegex(ValueError, "identity/coverage"):
                    check(rows)

    def test_independent_review_linkage_and_full_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            driver = scripts / "r1_3_kidrare_filter_full.py"
            tests = scripts / "test_r1_3_kidrare_filter_full.py"
            pilot_driver = scripts / "r1_3_kidrare_filter_extension.py"
            pilot_tests = scripts / "test_r1_3_kidrare_filter_extension.py"
            driver.write_text("driver")
            tests.write_text("tests")
            pilot_driver.write_text("pilot driver")
            pilot_tests.write_text("pilot tests")
            dest = root / s.full.DEST
            dest.mkdir()
            for name in ("preflight_receipt.json", "merge_receipt.json",
                         "COMPLETE.json"):
                (dest / name).write_text("{}")
            manifest = {"bindings": {
                name: {"path": str(path.resolve()), "sha256": s.digest(path)}
                for name, path in (("pilot_driver", pilot_driver),
                                   ("pilot_tests", pilot_tests),
                                   ("full_driver", driver), ("full_tests", tests))}}
            (dest / "manifest.json").write_text(json.dumps(manifest))
            (dest / "COMPLETE.json").write_text(json.dumps(
                {"status": "technical_full_extension_for_independent_review"}))
            review = {
                "status": "accepted_technical_full_extension_for_scientific_synthesis",
                "complete_sha256": s.digest(dest / "COMPLETE.json"),
                "manifest_sha256": s.digest(dest / "manifest.json"),
                "merge_receipt_sha256": s.digest(dest / "merge_receipt.json"),
                "reviewer": {"agent_id": s.CONTRACT.reviewer},
                "blocking_findings": [],
                "not_scientific_gate_c_approval": True,
            }
            (dest / "FULL_INDEPENDENT_REVIEW.json").write_text(json.dumps(review))
            spec = replace(s.CONTRACT, full_driver_sha=s.digest(driver),
                           full_tests_sha=s.digest(tests),
                           full_manifest_sha=s.digest(dest / "manifest.json"),
                           full_preflight_sha=s.digest(dest / "preflight_receipt.json"))
            with patch.object(s.pilot, "__file__", str(pilot_driver)), patch.object(
                    s.full, "__file__", str(driver)), patch.object(
                    s.full.Full, "verify_complete_chain") as verify:
                s.full_gate(root, spec)
                verify.assert_called_once()
                foreign = scripts / "foreign_pilot.py"
                foreign.write_text("pilot driver")
                with patch.object(s.pilot, "__file__", str(foreign)):
                    with self.assertRaisesRegex(ValueError, "path changed"):
                        s.full_gate(root, spec)
                pilot_driver.write_text("altered pilot driver")
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    s.full_gate(root, spec)
                pilot_driver.write_text("pilot driver")
                def mutate_during_verification():
                    pilot_driver.write_text("altered during full verification")
                verify.side_effect = mutate_during_verification
                with self.assertRaisesRegex(ValueError, "Consumed input changed"):
                    s.full_gate(root, spec)
                pilot_driver.write_text("pilot driver")
                verify.side_effect = None
                review["blocking_findings"] = ["unexpected"]
                (dest / "FULL_INDEPENDENT_REVIEW.json").write_text(
                    json.dumps(review))
                with self.assertRaisesRegex(ValueError, "acceptance"):
                    s.full_gate(root, spec)
                review["blocking_findings"] = []
                review["merge_receipt_sha256"] = "0" * 64
                (dest / "FULL_INDEPENDENT_REVIEW.json").write_text(
                    json.dumps(review))
                with self.assertRaisesRegex(ValueError, "acceptance"):
                    s.full_gate(root, spec)

    def test_tiny_output_and_refused_overwrite_or_input_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text('{"status":"accepted"}')
            a = analysis(["WT", "NB"])
            pop = {"stv": a.target, "kid": pd.DataFrame(
                {"slide_id": [], "pathpt_only_new_label": []}),
                "ccdi_prep": a.target}

            def gate(out, spec):
                if json.loads(source.read_text())["status"] != "accepted":
                    raise ValueError("Review no longer accepted")
                return {"review": source}

            def loader(out, spec, bindings):
                bindings["fixture"] = source
                for key in (
                    "correction_artifact_points_all_44_and_primary_intervals.csv",
                    "correction_artifact_paired_differences.json",
                    "ccdi_paired_differences.json",
                ):
                    bindings[key] = source
                return [a], {"fixture_report": {"sha256": s.digest(source)}}, pop

            dest = s.synthesize(root, s.Spec(b=10), gate=gate, loader=loader)
            self.assertTrue((dest / "COMPLETE.json").is_file())
            index = json.loads((dest / "provenance.json").read_text())["output_index"]
            self.assertEqual(set(index), set(s.OUTPUTS))
            self.assertEqual(len(pd.read_csv(dest / s.OUTPUTS["inventory"])), 2)
            with self.assertRaisesRegex(ValueError, "Refusing"):
                s.synthesize(root, s.Spec(b=10), gate=gate, loader=loader)

    def test_late_input_mutation_leaves_partial_without_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "review.json"
            source.write_text('{"status":"accepted"}')
            a = analysis(["WT", "NB"])
            def gate(out, spec):
                return {"review": source}
            def loader(out, spec, bindings):
                for key in ("correction_artifact_points_all_44_and_primary_intervals.csv",
                            "correction_artifact_paired_differences.json",
                            "ccdi_paired_differences.json"):
                    bindings[key] = source
                return [a], {}, {"stv": a.target, "kid": pd.DataFrame(
                    {"slide_id": [], "pathpt_only_new_label": []}),
                    "ccdi_prep": a.target}
            writer = s.output_frame
            def mutate(rows, path):
                writer(rows, path)
                source.write_text('{"status":"changed"}')
            with patch.object(s, "output_frame", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    s.synthesize(root, s.Spec(b=8), gate=gate, loader=loader)
            self.assertFalse((root / s.DEST / "COMPLETE.json").exists())
            source.write_text('{"status":"rejected"}')
            with self.assertRaisesRegex(ValueError, "Refusing"):
                s.synthesize(root, s.Spec(b=10), gate=gate, loader=loader)

    def test_final_gate_mutation_refuses_marker_for_input_and_outputs(self):
        for mutated in ("input", "broad", "provenance", "output_manifest"):
            with self.subTest(mutated=mutated), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "accepted_source.json"
                source.write_text('{"accepted":true}')
                a = analysis(["WT", "NB"])
                calls = [0]
                def gate(out, spec):
                    calls[0] += 1
                    if calls[0] == 3:
                        target = (source if mutated == "input" else
                                  root / s.DEST / (
                                      "output_manifest.json" if mutated == "output_manifest"
                                      else s.OUTPUTS[mutated]))
                        target.write_text(target.read_text() + " changed")
                    return {"review": source}
                def loader(out, spec, bindings):
                    for key in (
                        "correction_artifact_points_all_44_and_primary_intervals.csv",
                        "correction_artifact_paired_differences.json",
                        "ccdi_paired_differences.json"):
                        bindings[key] = source
                    return [a], {}, {"stv": a.target,
                                      "kid": pd.DataFrame({
                                          "slide_id": [], "pathpt_only_new_label": []}),
                                      "ccdi_prep": a.target}
                with self.assertRaisesRegex(ValueError, "changed"):
                    s.synthesize(root, s.Spec(b=8), gate=gate, loader=loader)
                self.assertEqual(calls[0], 3)
                self.assertTrue((root / s.DEST).is_dir())
                self.assertFalse((root / s.DEST / "COMPLETE.json").exists())

    def test_production_identity_contract_and_ccdi_filter_output_shapes(self):
        identities, sources, contrasts = s.production_identities()
        self.assertEqual((len(identities), len(sources), len(contrasts)),
                         (32, 44, 25))
        for name in ("query_tumor_ot", "both_tumor_ot"):
            self.assertIn(("ccdi", "ffpe_scope_clear", name), identities)
            self.assertIn(f"{name}_ffpe_add_all_unresolved", sources)
            self.assertIn(f"ccdi:ffpe_scope_clear:{name}:vs:whole_slide_ot",
                          contrasts)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "accepted.json"
            source.write_text('{"accepted":true}')
            configurations = [analysis(["WT", "NB"], cohort="ccdi",
                                       population_name="ffpe_scope_clear",
                                       configuration=name)
                              for name in ("whole_slide_ot", "query_tumor_ot",
                                           "both_tumor_ot")]
            def gate(out, spec):
                return {"review": source}
            def loader(out, spec, bindings):
                for key in (
                    "correction_artifact_points_all_44_and_primary_intervals.csv",
                    "correction_artifact_paired_differences.json",
                    "ccdi_paired_differences.json"):
                    bindings[key] = source
                return configurations, {}, {
                    "ccdi_prep": configurations[0].target,
                    "stv": configurations[0].target,
                    "kid": pd.DataFrame({
                        "slide_id": [], "pathpt_only_new_label": []})}
            dest = s.synthesize(root, s.Spec(b=8), gate=gate, loader=loader)
            inventory = pd.read_csv(dest / s.OUTPUTS["inventory"])
            paired = pd.read_csv(dest / s.OUTPUTS["paired"])
            self.assertEqual(len(inventory), 6)
            self.assertEqual(set(inventory.configuration),
                             {"whole_slide_ot", "query_tumor_ot", "both_tumor_ot"})
            self.assertEqual(set(paired.comparison_id), {
                "ccdi:ffpe_scope_clear:query_tumor_ot:vs:whole_slide_ot",
                "ccdi:ffpe_scope_clear:both_tumor_ot:vs:whole_slide_ot"})
            self.assertEqual(set(paired.tumor_group),
                             {"WT", "NB", "micro", "macro"})
            self.assertTrue((dest / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
