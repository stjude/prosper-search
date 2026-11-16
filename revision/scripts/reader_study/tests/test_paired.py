"""Constructed-example tests for the between-reader analysis (plan 002 §5.7)."""
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired as pr  # noqa: E402


class TestKappa(unittest.TestCase):
    def test_hand_computed_three_labels(self):
        # p_o = 5/8; marginals RMS 3, EWS 2, NB 3 for both; p_e = (9+4+9)/64; kappa = 3/7
        r1 = ["RMS", "RMS", "RMS", "EWS", "EWS", "NB", "NB", "NB"]
        r2 = ["RMS", "RMS", "EWS", "EWS", "NB", "NB", "NB", "RMS"]
        k, note = pr.cohen_kappa(r1, r2)
        self.assertAlmostEqual(k, 3 / 7, places=12)
        self.assertEqual(note, "")

    def test_hand_computed_two_labels(self):
        r1 = ["RMS"] * 5 + ["EWS"] * 5
        r2 = ["RMS"] * 4 + ["EWS"] * 5 + ["RMS"]
        self.assertAlmostEqual(pr.cohen_kappa(r1, r2)[0], 0.6, places=12)

    def test_perfect_and_disjoint(self):
        self.assertAlmostEqual(pr.cohen_kappa(["RMS", "NB"], ["RMS", "NB"])[0], 1.0)
        self.assertAlmostEqual(pr.cohen_kappa(["RMS", "RMS"], ["NB", "NB"])[0], 0.0)

    def test_one_label_and_empty(self):
        k, note = pr.cohen_kappa(["RMS"] * 4, ["RMS"] * 4)
        self.assertTrue(math.isnan(k))
        self.assertEqual(note, pr.KAPPA_ONE_LABEL)
        k, note = pr.cohen_kappa([], [])
        self.assertTrue(math.isnan(k))
        self.assertEqual(note, pr.KAPPA_EMPTY)

    def test_off_list_folds_into_misc(self):
        # 003 Amendment 006 item 4: free-text labels count as MISC (other) for kappa; note records the count.
        # Hand-computed: r1 = [RMS, RMS, MISC, EWS], r2 = [RMS, MISC, MISC, EWS] -> p_o = 3/4;
        # marginals r1 RMS 2, MISC 1, EWS 1; r2 RMS 1, MISC 2, EWS 1 -> p_e = (2+2+1)/16 = 5/16; k = 7/11
        k, note = pr.cohen_kappa(["RMS", "RMS", "Rhabdo", "EWS"], ["RMS", "MISC (other)", "kikuchi", "EWS"])
        self.assertAlmostEqual(k, 7 / 11, places=12)
        self.assertEqual(note, "2 off-list label(s) folded into MISC (other)")
        k, note = pr.cohen_kappa(["Rhabdo", "SFT"], ["MISC (other)", "LMS"])
        self.assertTrue(math.isnan(k))
        self.assertTrue(note.startswith(pr.KAPPA_ONE_LABEL))

    def test_rejects_length(self):
        with self.assertRaises(ValueError):
            pr.cohen_kappa(["RMS"], [])


class TestAgreement(unittest.TestCase):
    def test_counts(self):
        s = pr.agreement_summary(["RMS", "NB", "EWS", "OS"], ["RMS", "WT", "EWS", "NB"],
                                 [True, True, False, False], [True, False, True, False])
        self.assertEqual((s["n"], s["n_agree"], s["raw_agreement"]), (4, 2, 0.5))
        self.assertEqual((s["both_correct"], s["only_reader1_correct"], s["only_reader2_correct"],
                          s["neither_correct"]), (1, 1, 1, 1))

    def test_empty(self):
        s = pr.agreement_summary([], [], [], [])
        self.assertEqual((s["n"], s["n_agree"], s["raw_agreement"], s["both_correct"]), (0, 0, None, 0))
        self.assertTrue(math.isnan(s["cohen_kappa"]))
        self.assertEqual(s["kappa_note"], pr.KAPPA_EMPTY)

    def test_paired_case_ids(self):
        self.assertEqual(pr.paired_case_ids(["c3", "c1", "c2"], ["c2", "c4", "c3"]), ["c2", "c3"])
        self.assertEqual(pr.paired_case_ids(["c1"], ["c2"]), [])


def _rows(reader, spec):
    out = []
    for cid, favs, cors, cls in spec:
        row = {"reader_id": reader, "case_id": cid, "cohort": "CCDI", "reference": "RMS", "hit5": True,
               "class_AB": cls}
        for p, f, c in zip("ABC", favs, cors):
            row[f"favoured_{p}"] = f
            row[f"fav_correct_{p}"] = c
        out.append(row)
    return out


class TestTables(unittest.TestCase):
    def test_between_reader_tables(self):
        r = pd.DataFrame(
            _rows("a", [("c1", ("RMS", "RMS", "RMS"), (True, True, True), "confirm"),
                        ("c2", ("NB", "RMS", "RMS"), (False, True, True), "correct"),
                        ("c3", ("EWS", "EWS", None), (False, False, None), "unhelpful")])
            + _rows("b", [("c1", ("NB", "RMS", "RMS"), (False, True, True), "correct"),
                          ("c3", ("EWS", "EWS", "EWS"), (False, False, False), "unhelpful"),
                          ("c9", ("RMS", "RMS", "RMS"), (True, True, True), "confirm")]))
        s, c = pr.between_reader_tables(r, ["a", "b"], {"a": "expert", "b": "trainee"})
        self.assertEqual(list(c.case_id), ["c1", "c3"])
        self.assertEqual(set(s.paired_set_n), {2})
        a = s.set_index("step")
        self.assertEqual((a.at["A", "n"], a.at["A", "n_agree"]), (2, 1))
        self.assertEqual((a.at["A", "only_reader1_correct"], a.at["A", "neither_correct"]), (1, 1))
        self.assertEqual((a.at["B", "n_agree"], a.at["B", "both_correct"]), (2, 1))
        self.assertEqual(a.at["C", "n"], 1)          # c3 lacks reader a's Step 3 answer
        self.assertEqual(a.at["C", "kappa_note"], pr.KAPPA_ONE_LABEL)
        self.assertEqual(list(c.columns[-7:]), ["r2_favoured_A", "r2_favoured_B", "r2_favoured_C",
                                                "r2_fav_correct_A", "r2_fav_correct_B", "r2_fav_correct_C",
                                                "r2_class_AB"])
        self.assertEqual(c.set_index("case_id").at["c1", "r2_class_AB"], "correct")

    def test_no_overlap(self):
        r = pd.DataFrame(_rows("a", [("c1", ("RMS",) * 3, (True,) * 3, "confirm")])
                         + _rows("b", [("c2", ("RMS",) * 3, (True,) * 3, "confirm")]))
        s, c = pr.between_reader_tables(r, ["a", "b"], {"a": "expert", "b": "expert"})
        self.assertTrue(c.empty)
        self.assertEqual(list(s.n), [0, 0, 0])
        self.assertTrue(all(s.kappa_note == pr.KAPPA_EMPTY))


if __name__ == "__main__":
    unittest.main()
