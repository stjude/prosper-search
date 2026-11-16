"""Constructed-example tests for the endpoint helpers (plan 002 §4–§6)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import endpoints as ep  # noqa: E402


class TestWilson(unittest.TestCase):
    def test_known_values(self):
        p, lo, hi = ep.wilson(5, 10)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(lo, 0.236593, places=5)
        self.assertAlmostEqual(hi, 0.763407, places=5)
        p, lo, hi = ep.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.277533, places=5)
        p, lo, hi = ep.wilson(10, 10)
        self.assertAlmostEqual(lo, 0.722467, places=5)
        self.assertAlmostEqual(hi, 1.0, places=12)
        self.assertEqual(ep.wilson(0, 0), (None, None, None))


class TestBootstrap(unittest.TestCase):
    def test_weights_shape_and_seed(self):
        w = ep.case_weights(80)
        self.assertEqual(w.shape, (5000, 80))
        self.assertTrue((w.sum(axis=1) == 80).all())
        self.assertTrue((w == ep.case_weights(80)).all())
        self.assertFalse((w == ep.case_weights(80, seed=1)).all())

    def test_ratio_constant_and_estimate(self):
        cases = [f"c{i}" for i in range(20)]
        w = ep.case_weights(len(cases), n_draws=500)
        r = ep.proportion_endpoint(cases, [True] * 20, cases, w)
        self.assertEqual((r["k"], r["n"], r["estimate"], r["boot95_lo"], r["boot95_hi"]), (20, 20, 1.0, 1.0, 1.0))
        flags = [i < 10 for i in range(20)]
        r = ep.proportion_endpoint(cases, flags, cases, w)
        self.assertEqual(r["estimate"], 0.5)
        self.assertLess(r["boot95_lo"], 0.5)
        self.assertGreater(r["boot95_hi"], 0.5)
        self.assertAlmostEqual(r["wilson95_lo"], ep.wilson(10, 20)[1])

    def test_paired_difference_within_draw(self):
        cases = [f"c{i}" for i in range(30)]
        w = ep.case_weights(len(cases), n_draws=400)
        a = [i % 2 == 0 for i in range(30)]
        r = ep.paired_difference(cases, a, a, cases, w)
        self.assertEqual((r["estimate"], r["boot95_lo"], r["boot95_hi"]), (0.0, 0.0, 0.0))
        # B = A shifted for 6 cases: 3 wrong→right, 0 right→wrong
        b = list(a)
        for i in (1, 3, 5):
            b[i] = True
        r = ep.paired_difference(cases, a, b, cases, w)
        self.assertAlmostEqual(r["estimate"], 0.1)
        self.assertEqual((r["x_wrong_y_right"], r["x_right_y_wrong"]), (3, 0))
        self.assertGreaterEqual(r["boot95_lo"], 0.0)
        # an independent-margins bootstrap would give negative lower bounds here; within-draw cannot
        self.assertTrue(r["boot95_hi"] > r["estimate"])

    def test_two_readers_enter_together(self):
        cases = [f"c{i}" for i in range(15)]
        ids = cases + cases
        flags = [True] * 15 + [False] * 15
        w = ep.case_weights(len(cases), n_draws=300)
        r = ep.proportion_endpoint(ids, flags, cases, w)
        self.assertEqual((r["estimate"], r["boot95_lo"], r["boot95_hi"]), (0.5, 0.5, 0.5))

    def test_per_case_alignment(self):
        num, den = ep.per_case(["b", "a", "b"], [1, 0, 1], ["a", "b"])
        self.assertEqual(num.tolist(), [0.0, 2.0])
        self.assertEqual(den.tolist(), [1.0, 2.0])


class TestTables(unittest.TestCase):
    def test_class_table(self):
        classes = ["confirm", "confirm", "correct", "mislead", "confirm", "unhelpful"]
        subs = ["confirm_favoured", "confirm_differential_only", "", "", "confirm_favoured", ""]
        rows = ep.class_table(classes, subs)
        got = {r["class"].strip(): r["count"] for r in rows}
        self.assertEqual(got, {"mislead": 1, "correct": 1, "expand": 0, "confirm": 3, "confirm_favoured": 2,
                               "confirm_differential_only": 1, "unhelpful": 1, "other change": 0})
        self.assertEqual([r["class"].strip() for r in rows][:5],
                         ["mislead", "correct", "expand", "confirm", "confirm_favoured"])
        self.assertAlmostEqual(rows[0]["proportion"], 1 / 6)
        with self.assertRaises(ValueError):
            ep.class_table(["bogus"], [""])

    def test_likert(self):
        s = ep.likert_summary([1, 2, 2, 5, None, float("nan")])
        self.assertEqual((s["n_1"], s["n_2"], s["n_3"], s["n_4"], s["n_5"], s["n"]), (1, 2, 0, 0, 1, 4))
        self.assertEqual(s["median"], 2.0)
        with self.assertRaises(ValueError):
            ep.likert_summary([0, 3])
        self.assertIsNone(ep.likert_summary([])["median"])

    def test_median_iqr(self):
        s = ep.median_iqr([10, 20, 30, 40, None])
        self.assertEqual((s["n"], s["median"], s["q1"], s["q3"]), (4, 25.0, 17.5, 32.5))

    def test_group_tertiles(self):
        counts = {"DSRCT": 31, "MRT": 41, "NF": 45, "MPNST": 48, "MEL": 76, "LIVER": 140, "SS": 142,
                  "ACT/ACC": 149, "RB": 191, "EWS": 217, "OS": 258, "WT": 368, "NB": 378, "RMS": 469}
        t = ep.group_tertiles(counts)
        self.assertEqual(sorted(g for g, v in t.items() if v == "low"), ["DSRCT", "MEL", "MPNST", "MRT", "NF"])
        self.assertEqual(sorted(g for g, v in t.items() if v == "mid"), ["ACT/ACC", "LIVER", "RB", "SS"])
        self.assertEqual(sorted(g for g, v in t.items() if v == "high"), ["EWS", "NB", "OS", "RMS", "WT"])

    def test_chrono_quartile(self):
        self.assertEqual([ep.chrono_quartile(r, 80) for r in (1, 20, 21, 60, 61, 80)], [1, 1, 2, 3, 4, 4])

    def test_negation_only(self):
        self.assertTrue(ep.discrepancy_negation_only("Six are RMS. No discrepant neighbors were identified."))
        self.assertTrue(ep.discrepancy_negation_only("All neighbors are pediatric; none are flagged as discrepant."))
        self.assertTrue(ep.discrepancy_negation_only("All are pediatric, and there are no discrepant neighbors."))
        self.assertFalse(ep.discrepancy_negation_only(
            "Two SS neighbors are flagged discrepant [x]; no RMS neighbors are discrepant."))
        self.assertFalse(ep.discrepancy_negation_only(
            "The NB neighbors are discrepant; none carry an internal discrepant flag."))
        self.assertFalse(ep.discrepancy_negation_only("Votes are split between NB and OS."))
        self.assertFalse(ep.discrepancy_negation_only("Overall the evidence supports RMS."))


if __name__ == "__main__":
    unittest.main()
