"""Constructed-example tests for the tolerant LLM-summary parser (002 §3.8, 003 §B(4))."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_parse as lp  # noqa: E402


def scores(text):
    s, unplaced = lp.parse_scores(text)
    return s, unplaced


class TestScores(unittest.TestCase):
    def test_list_form(self):
        s, u = scores("Bayesian disease probabilities are WT 0.427, MPNST 0.296, NF 0.196, and SS 0.081. "
                      "By votes, MPNST and WT are co-dominant (4 each); the Bayesian model favors WT at 0.427.")
        self.assertEqual(s, {"WT": 0.427, "MPNST": 0.296, "NF": 0.196, "SS": 0.081})
        self.assertEqual(u, [])

    def test_parenthesised_and_slash_code(self):
        s, _ = scores("Bayesian disease probabilities favor EWS (0.752), followed by ACT/ACC (0.100), WT (0.050).")
        self.assertEqual(s, {"EWS": 0.752, "ACT/ACC": 0.1, "WT": 0.05})

    def test_p_notation_without_probability_word(self):
        t = "This yields a dominant diagnosis of RMS (10 votes), and the Bayesian model assigns P(RMS)=1.00."
        self.assertEqual(scores(t)[0], {"RMS": 1.0})
        self.assertTrue(lp.parse_summary(t)["llm_wording_probability"])

    def test_percent_and_number_before_group(self):
        s, _ = scores("The Bayesian model assigns 93.1% probability to RMS and 6.9% to EWS.")
        self.assertEqual(s, {"RMS": 0.931, "EWS": 0.069})
        s, _ = scores("Bayesian tumor-group probabilities favor RMS 30.2% (highest), then LIVER 27.4%, MEL 13.3%.")
        self.assertEqual(s, {"RMS": 0.302, "LIVER": 0.274, "MEL": 0.133})
        s, _ = scores("The Bayesian model assigns 1.000 probability to the ACT/ACC tumor group.")
        self.assertEqual(s, {"ACT/ACC": 1.0})

    def test_value_then_for_group(self):
        s, _ = scores("The dominant diagnosis is SS (10 votes), and the Bayesian probability is 1.0 for SS.")
        self.assertEqual(s, {"SS": 1.0})
        s, _ = scores("This yields 9 NB votes and, together with a Bayesian probability of 1.00 for NB, supports NB.")
        self.assertEqual(s, {"NB": 1.0})

    def test_group_only_earlier_in_sentence(self):
        s, _ = scores("RB is the dominant diagnosis (10 votes), and the Bayesian probability is 1.00.")
        self.assertEqual(s, {"RB": 1.0})
        s, _ = scores("A Bayesian model assigns NB a probability of 1.00, which contrasts with the neighbor "
                      "vote favoring ganglioneuroblastoma.")
        self.assertEqual(s, {"NB": 1.0})
        s, _ = scores("Bayesian disease probabilities indicate RB=1.0.")
        self.assertEqual(s, {"RB": 1.0})

    def test_full_names(self):
        s, _ = scores("By votes, neuroblastoma is the mode diagnosis (3 of 10), but Bayesian probabilities modestly "
                      "favor melanoma (0.321) over neuroblastoma (0.312), with lower support for rhabdomyosarcoma "
                      "(0.172), synovial sarcoma (0.141), and Ewing sarcoma (0.054).")
        self.assertEqual(s, {"MEL": 0.321, "NB": 0.312, "RMS": 0.172, "SS": 0.141, "EWS": 0.054})
        s, _ = scores("Bayesian disease probabilities favor neuroblastoma (NB) at 1.0.")
        self.assertEqual(s, {"NB": 1.0})

    def test_non_score_numbers_ignored(self):
        s, u = scores("The patient is a 0.2-year-old male. Bayesian probabilities favor SS at 0.652 over RMS at 0.348.")
        self.assertEqual(s, {"SS": 0.652, "RMS": 0.348})
        self.assertEqual(u, [])

    def test_sentences_without_score_wording_ignored(self):
        s, _ = scores("The age (3.3 years) matches none. Four are WT 0.5 by eye.")
        self.assertEqual(s, {})


class TestLeading(unittest.TestCase):
    def lead(self, t):
        return lp.leading_groups(t)[:2]

    def test_alternative_is_not_leading(self):
        g, src = self.lead("Overall the neighbor and Bayesian evidence supports SS as the leading consideration, "
                           "with EWS as the main alternative.")
        self.assertEqual((g, src), (["SS"], "leading_sentence"))

    def test_multiple_leading(self):
        g, _ = self.lead("Overall the neighbor evidence is split between NB and OS by vote, while the Bayesian model "
                         "favors DSRCT; this supports NB, OS, and DSRCT as leading considerations.")
        self.assertEqual(g, ["NB", "OS", "DSRCT"])

    def test_is_the_leading(self):
        g, _ = self.lead("Overall, RMS is the leading consideration by neighbor votes (4 of 10) and by Bayesian "
                         "probability (0.522), while SS is a notable alternative.")
        self.assertEqual(g, ["RMS"])

    def test_subtype_codes(self):
        self.assertEqual(self.lead("Overall the neighbor evidence supports HB as the leading consideration "
                                   "within a LIVER tumor-group context.")[0], ["LIVER"])
        self.assertEqual(self.lead("Overall the neighbor evidence supports ACT as the leading consideration "
                                   "within the ACT/ACC spectrum.")[0], ["ACT/ACC"])
        self.assertEqual(self.lead("Overall, the neighbor evidence supports Ewing sarcoma as the leading "
                                   "consideration.")[0], ["EWS"])

    def test_overall_fallback(self):
        g, src = self.lead("Overall the neighbor evidence supports a WT–MPNST differential with a modest Bayesian "
                           "preference for WT.")
        self.assertEqual((g, src), (["WT", "MPNST"], "overall_sentence"))
        g, _ = self.lead("Overall the neighbor evidence supports an RMS-predominant pattern while acknowledging "
                         "liver and melanoma as important alternatives.")
        self.assertEqual(g, ["RMS"])
        g, _ = self.lead("Overall, neighbor votes are split between NB and RMS (3 each), whereas the Bayesian model "
                         "slightly favors RMS, with additional support for MPNST and NB; this pattern suggests "
                         "considering neuroblastoma and rhabdomyosarcoma foremost while keeping MPNST and DSRCT "
                         "in the differential.")
        self.assertEqual(g, ["NB", "RMS"])

    def test_leading_in_score_sentence(self):
        g, _ = self.lead("Bayesian disease probabilities are SS 0.882 and OS 0.118, supporting SS as the leading "
                         "consideration. The thorax primary site matches none of the neighbors.")
        self.assertEqual(g, ["SS"])

    def test_none(self):
        self.assertEqual(self.lead("The liver site matches none of the neighbors."), ([], "none"))


class TestCaseVars(unittest.TestCase):
    def test_truth_unlisted_and_flags(self):
        t = ("Among 10 neighbor cases, six are RMS. Bayesian disease probabilities are RMS 0.6 and EWS 0.4. "
             "No discrepant neighbors were identified. Overall the neighbor evidence supports RMS as the "
             "leading consideration.")
        v = lp.llm_case_vars(lp.parse_summary(t), "SS")
        self.assertEqual(v["llm_top"], "RMS")
        self.assertFalse(v["llm_top_correct"])
        self.assertEqual(v["llm_truth_prob"], 0.0)
        self.assertFalse(v["llm_names_truth_leading"])
        self.assertTrue(v["llm_flags_discrepancy"])      # literal term match, negation included
        self.assertTrue(v["llm_wording_probability"])
        v = lp.llm_case_vars(lp.parse_summary(t), "EWS")
        self.assertEqual(v["llm_truth_prob"], 0.4)

    def test_tie_and_literal_terms(self):
        t = "Bayesian disease probabilities are RMS 0.5 and EWS 0.5. The pattern shows some discrepancy."
        p = lp.parse_summary(t)
        self.assertEqual(p["llm_top"], "EWS|RMS")
        self.assertTrue(p["llm_top_tie"])
        self.assertFalse(lp.llm_case_vars(p, "RMS")["llm_top_correct"])
        self.assertFalse(lp.llm_case_vars(p, "RMS")["llm_flags_discrepancy"])
        self.assertTrue(lp.parse_summary("Votes are split.")["discrepancy_terms"] == ["split"])

    def test_no_scores(self):
        p = lp.parse_summary("Overall the neighbor evidence supports NB as the leading consideration.")
        v = lp.llm_case_vars(p, "NB")
        self.assertIsNone(v["llm_top"])
        self.assertIsNone(v["llm_truth_prob"])
        self.assertFalse(v["llm_wording_probability"])
        self.assertTrue(v["llm_names_truth_leading"])


if __name__ == "__main__":
    unittest.main()
