"""Focused tests for v3's objective, strict quotas and post-selection audit."""
import contextlib
import io
import itertools
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

os.environ.setdefault("PROSPER_OUT", str(Path(__file__).resolve().parents[2] / "prosper_revision_output"))
from build_review_pool import selection_checks, solve_pool


class ReviewPoolTests(unittest.TestCase):
    def setUp(self):
        self.pop = pd.DataFrame({
            "tumor_group": ["A", "A", "B", "B", "C", "C"],
            "patient_id": list("abcdef"),
        }, index=list("abcdef"))
        self.ali = pd.DataFrame({
            "selection_tier": ["Tier1", "Tier1", "Tier2"],
            "tier": ["difficult", "moderate", "no_model_signal"],
        }, index=list("ace"))
        self.nb = {"a": {"b", "c"}, "b": {"a", "d"}, "c": {"d", "e"},
                   "d": {"c", "f"}, "e": {"f", "a"}, "f": {"e", "b"}}
        self.eligible = set(self.pop.index)
        self.args = SimpleNamespace(budget=3, ali_quota_frac=.5, hard_min=1,
                                    moderate_min=1, nosignal_min=1, ali_easy_max=0,
                                    class_max_frac=.7)

    def solve(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return solve_pool(self.pop, self.nb, self.eligible, self.ali, self.args)

    def test_strict_mix_and_ceil_quota(self):
        selected, covered, checks = self.solve()
        self.assertEqual(set(selected), set("ace"))
        self.assertEqual(covered, set("abcdef"))
        self.assertTrue(checks.passed.all())
        self.assertEqual(checks.set_index("constraint").loc["ali_tier1", "minimum"], 2)

    def test_maximum_coverage_matches_exhaustive_search(self):
        self.args.hard_min = self.args.moderate_min = self.args.nosignal_min = 0
        self.eligible.remove("b")
        _, covered, _ = self.solve()
        scores = []
        for selected in itertools.combinations(sorted(self.eligible), self.args.budget):
            if selection_checks(selected, self.pop, self.eligible, self.ali, self.args).passed.all():
                scores.append(len(set().union(*(self.nb[s] | {s} for s in selected))))
        self.assertEqual(len(covered), max(scores))

    def test_unavailable_no_signal_is_not_silently_relaxed(self):
        self.ali = self.ali.drop("e")
        with self.assertRaisesRegex(ValueError, "Infeasible"):
            self.solve()
        self.args.nosignal_min = 0
        self.assertTrue(self.solve()[2].passed.all())

    def test_ineligible_class_is_not_silently_omitted(self):
        self.args.nosignal_min = 0
        self.eligible -= set("ef")
        with self.assertRaisesRegex(ValueError, "Infeasible"):
            self.solve()

    def test_ali_cannot_bypass_tile_eligibility(self):
        self.eligible.remove("a")
        with self.assertRaisesRegex(ValueError, "Infeasible"):
            self.solve()

    def test_patient_limit_is_hard(self):
        self.pop.loc["c", "patient_id"] = "a"
        with self.assertRaisesRegex(ValueError, "Infeasible"):
            self.solve()

    def test_class_cap_is_hard(self):
        self.args.class_max_frac = .3
        with self.assertRaisesRegex(ValueError, "Infeasible"):
            self.solve()

    def test_coverage_edge_direction(self):
        self.pop = self.pop.loc[list("abcd")].copy()
        self.pop["tumor_group"] = "A"
        self.ali = self.ali.loc[["a"]]
        self.nb = {"a": {"b"}, "b": {"a"}, "c": {"a"}, "d": {"a"}}
        self.args = SimpleNamespace(budget=1, ali_quota_frac=1, hard_min=0,
                                    moderate_min=0, nosignal_min=0, ali_easy_max=1,
                                    class_max_frac=1)
        selected, covered, _ = self.solve()
        self.assertEqual(selected, ["a"])
        self.assertEqual(covered, {"a", "b"})

    def test_manual_swap_is_rechecked(self):
        checks = selection_checks(list("acd"), self.pop, self.eligible, self.ali, self.args)
        failed = set(checks.loc[~checks.passed, "constraint"])
        self.assertIn("ali_no_signal", failed)
        self.assertIn("class:C", failed)


if __name__ == "__main__":
    unittest.main()
