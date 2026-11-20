import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import search_vs_reader as sv  # noqa: E402


class TestRules(unittest.TestCase):
    def test_plurality_tie_goes_to_nearest_ranked_tied_label(self):
        labels = ["OS", "RMS", "EWS", "RMS", "EWS", "EWS", "EWS", "NB", "NB", "NB"]
        self.assertEqual(sv.plurality(labels, 5), "RMS")      # RMS 2, EWS 2: RMS is ranked nearer
        self.assertEqual(sv.plurality(labels, 10), "EWS")     # EWS 4
        self.assertEqual(sv.plurality(labels, 1), "OS")
        self.assertEqual(sv.plurality(["OS", "RMS", "EWS", "NB", "WT"], 5), "OS")  # all tied: top-1

    def test_consensus_requires_a_single_leader(self):
        self.assertEqual(sv.consensus(["RMS", "RMS", "EWS", "NB", "OS"]), "RMS")
        self.assertEqual(sv.consensus(["RMS", "RMS", "EWS", "EWS", "OS"]), "")

    def test_mcnemar_exact(self):
        self.assertEqual(sv.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(sv.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(sv.mcnemar_exact(0, 6), 2 / 64)
        self.assertAlmostEqual(sv.mcnemar_exact(1, 9), 2 * 11 / 1024)

    def test_compare_counts_and_sign(self):
        cases = ["c1", "c2", "c3", "c4"]
        e = sv.compare(cases, [True, False, False, True], [True, True, True, False])
        self.assertEqual((e["reader_correct"], e["search_correct"]), (2, 3))
        self.assertEqual((e["both_right"], e["only_search_right"], e["only_reader_right"], e["neither_right"]),
                         (1, 2, 1, 0))
        self.assertAlmostEqual(e["diff_search_minus_reader"], 0.25)


if __name__ == "__main__":
    unittest.main()
