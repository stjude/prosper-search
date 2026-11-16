"""Constructed-example tests for plan 002 §3 (as amended by 003 §B). No real readings are used."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import derive as d  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
COMMITTED_MAP = REPO / "docs" / "pathologist_study" / "reader_study_misc_entity_map_20261027.csv"


def A(dx, fav, phase="A"):
    return d.make_answer(dx, fav, phase)


class TestEffectiveAnswer(unittest.TestCase):
    def test_make_answer_compacts_and_validates(self):
        a = A(["RMS", "", " EWS "], "EWS")
        self.assertEqual(a.dx, ("RMS", "EWS"))
        with self.assertRaises(ValueError):
            A(["RMS"], "EWS")
        with self.assertRaises(ValueError):
            A(["RMS", "rms"], "RMS")
        with self.assertRaises(ValueError):
            A(["", ""], "")

    def test_carry_forward(self):
        a = A(["RMS", "EWS"], "RMS")
        eff = d.effective_answers(a, "no", None, "no", None)
        self.assertIs(eff["B"], a)
        self.assertIs(eff["C"], a)

    def test_change_at_b_carries_to_c(self):
        a = A(["RMS"], "RMS")
        b = A(["EWS", "RMS"], "EWS", "B")
        eff = d.effective_answers(a, "yes", b, "no", None)
        self.assertIs(eff["B"], b)
        self.assertIs(eff["C"], b)
        self.assertEqual(eff["C"].source_phase, "B")

    def test_change_at_c_only(self):
        a = A(["RMS"], "RMS")
        c = A(["SS"], "SS", "C")
        eff = d.effective_answers(a, "no", None, "yes", c)
        self.assertIs(eff["B"], a)
        self.assertIs(eff["C"], c)

    def test_missing_c(self):
        a = A(["RMS"], "RMS")
        self.assertIsNone(d.effective_answers(a, "no", None, None, None)["C"])

    def test_inconsistent_payloads_raise(self):
        a = A(["RMS"], "RMS")
        with self.assertRaises(ValueError):
            d.effective_answers(a, "yes", None, "no", None)
        with self.assertRaises(ValueError):
            d.effective_answers(a, "no", A(["EWS"], "EWS", "B"), "no", None)
        with self.assertRaises(ValueError):
            d.effective_answers(a, "", None, "no", None)


class TestMatching(unittest.TestCase):
    def test_strict_match_uses_position_not_dx1(self):
        s = d.score_answer(A(["EWS", "RMS", "SS"], "RMS"), "RMS")
        self.assertEqual(s, {"fav_correct": True, "in_diff": True, "truth_rank": 2, "n_dx": 3})
        s = d.score_answer(A(["EWS", "RMS"], "EWS"), "RMS")
        self.assertEqual((s["fav_correct"], s["in_diff"], s["truth_rank"]), (False, True, 2))
        s = d.score_answer(A(["EWS"], "EWS"), "RMS")
        self.assertEqual((s["fav_correct"], s["in_diff"], s["truth_rank"]), (False, False, None))

    def test_misc_and_normal_never_match(self):
        with self.assertRaises(ValueError):
            d.score_answer(A([d.MISC], d.MISC), d.MISC)
        with self.assertRaises(ValueError):
            d.score_answer(A(["NORMAL"], "NORMAL"), "NORMAL")
        s = d.score_answer(A([d.MISC, "NORMAL"], d.MISC), "LIVER")
        self.assertFalse(s["in_diff"])

    def test_case_sensitive_exact(self):
        self.assertFalse(d.score_answer(A(["rms"], "rms"), "RMS")["in_diff"])


class TestEntityMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        self.tmp.write(
            "entity_as_written,normalised_entity,mapped_group,basis\n"
            '"embryonal sarcoma","Embryonal sarcoma",LIVER,"x"\n'
            '"anaplastic sarcoma of the kidney","Anaplastic sarcoma of kidney",outside_vocabulary,"x"\n'
            '"esthesioneuroblastoma / olfactory neuroblastoma","Esthesioneuroblastoma",outside_vocabulary,"x"\n'
            '"spindle cell/sclerosing rhabdomyosarcoma","RMS (spindle/sclerosing)",RMS,"x"\n')
        self.tmp.close()
        self.m = d.load_entity_map(Path(self.tmp.name))

    def tearDown(self):
        Path(self.tmp.name).unlink()

    def test_committed_map_loads(self):
        rows = d.load_entity_map(COMMITTED_MAP)
        self.assertEqual({r.mapped_group for r in rows}, {"LIVER", "RMS", "outside_vocabulary"})

    def test_mapped_entity_replaces_misc(self):
        a = A(["LIVER", d.MISC], d.MISC)
        out, note = d.apply_entity_map(a, "Favour undifferentiated Embryonal  Sarcoma of liver.", self.m)
        self.assertEqual(out.dx, ("LIVER", "LIVER"))
        self.assertEqual(out.favoured, "LIVER")
        self.assertTrue(note.startswith("misc_mapped"))
        self.assertTrue(d.score_answer(out, "LIVER")["fav_correct"])
        self.assertEqual(d.score_answer(out, "LIVER")["truth_rank"], 1)

    def test_outside_vocabulary_stays_unmatched(self):
        a = A([d.MISC, "WT"], "WT")
        out, note = d.apply_entity_map(a, "could be anaplastic sarcoma of the kidney", self.m)
        self.assertIs(out, a)
        self.assertTrue(note.startswith("misc_outside_vocabulary"))
        out, note = d.apply_entity_map(a, "olfactory neuroblastoma?", self.m)
        self.assertTrue(note.startswith("misc_outside_vocabulary"))

    def test_conflicting_entities_stay_unmatched(self):
        a = A([d.MISC], d.MISC)
        out, note = d.apply_entity_map(a, "embryonal sarcoma vs anaplastic sarcoma of the kidney", self.m)
        self.assertIs(out, a)
        self.assertTrue(note.startswith("misc_ambiguous"))

    def test_no_entity_or_no_misc(self):
        a = A([d.MISC], d.MISC)
        self.assertEqual(d.apply_entity_map(a, "unusual", self.m), (a, "misc_no_named_entity"))
        b = A(["RMS"], "RMS")
        self.assertEqual(d.apply_entity_map(b, "embryonal sarcoma", self.m), (b, ""))

    def test_no_false_hit_on_embryonal_rms(self):
        a = A([d.MISC], d.MISC)
        _, note = d.apply_entity_map(a, "embryonal rhabdomyosarcoma", self.m)
        self.assertEqual(note, "misc_no_named_entity")

    def test_rms_subtype_verbatim(self):
        a = A([d.MISC], d.MISC)
        out, _ = d.apply_entity_map(a, "Spindle cell/sclerosing rhabdomyosarcoma", self.m)
        self.assertEqual(out.favoured, "RMS")


class TestOutcomeClass(unittest.TestCase):
    def test_table(self):
        oc = d.outcome_class
        # fav_x, in_diff_x, fav_y, in_diff_y, changed
        self.assertEqual(oc(True, True, False, True, True), ("mislead", ""))     # favoured lost
        self.assertEqual(oc(False, True, False, False, True), ("mislead", ""))   # truth left differential
        self.assertEqual(oc(True, True, False, False, True), ("mislead", ""))
        self.assertEqual(oc(False, True, True, True, True), ("correct", ""))
        self.assertEqual(oc(False, False, True, True, True), ("correct", ""))    # enters as favoured
        self.assertEqual(oc(False, False, False, True, True), ("expand", ""))
        self.assertEqual(oc(True, True, True, True, False), ("confirm", "confirm_favoured"))
        self.assertEqual(oc(False, True, False, True, False), ("confirm", "confirm_differential_only"))
        self.assertEqual(oc(False, False, False, False, False), ("unhelpful", ""))
        # yes with an identical-scoring answer is "other change", never confirm
        self.assertEqual(oc(True, True, True, True, True), ("other change", ""))
        self.assertEqual(oc(False, False, False, False, True), ("other change", ""))
        self.assertEqual(oc(False, True, False, True, True), ("other change", ""))


class TestConfidenceAndTime(unittest.TestCase):
    def test_harmful_confidence(self):
        self.assertTrue(d.confidence_vars(3, 4, 4, False)["harmful_conf_B"])
        self.assertFalse(d.confidence_vars(3, 4, 4, True)["harmful_conf_B"])
        self.assertFalse(d.confidence_vars(3, 3, None, False)["harmful_conf_B"])
        v = d.confidence_vars(2, 4, 3, True)
        self.assertEqual((v["dconf_BA"], v["dconf_CB"]), (2, -1))
        self.assertIsNone(d.confidence_vars(2, 4, None, True)["dconf_CB"])

    def test_reading_time_and_interruptions(self):
        t = {"A": (0.0, 1800.0), "B": (1800.5, 3700.0), "C": (3700.1, 3800.0)}
        v = d.reading_time_vars(t)
        self.assertEqual(v["sec_A"], 1800.0)
        self.assertFalse(v["interrupted_A"])            # boundary: > 1800 only
        self.assertTrue(v["interrupted_B"])
        self.assertFalse(v["interrupted_between_phases"])
        t2 = {"A": (0.0, 10.0), "B": (1811.0, 1820.0), "C": (1821.0, 1830.0)}
        v2 = d.reading_time_vars(t2)
        self.assertAlmostEqual(v2["gap_AB_s"], 1801.0)
        self.assertTrue(v2["interrupted_between_phases"])
        t3 = {"A": (0.0, 10.0), "B": (11.0, 20.0), "C": (1821.0, 1830.0)}
        self.assertTrue(d.reading_time_vars(t3)["interrupted_between_phases"])
        v4 = d.reading_time_vars({"A": (0.0, 10.0), "B": (11.0, 20.0)})
        self.assertIsNone(v4["sec_C"])
        self.assertFalse(v4["interrupted_between_phases"])


class TestEngagement(unittest.TestCase):
    T = dict(t_submit_a=100.0, t_open_b=101.0, t_submit_b=200.0, t_open_c=201.0, t_submit_c=300.0)

    def ev(self, kind, ts, **payload):
        return {"kind": kind, "ts": ts, "payload": payload}

    def test_auto_excluded_and_windows(self):
        evs = [
            self.ev("neighbours_panel_toggled", 101.0, open=True, auto=True),
            self.ev("roi_selected", 150.0, roi_idx=3, via="click"),
            self.ev("hit_opened", 150.0, rank=1, auto=True, via="click"),   # auto-opened after ROI
            self.ev("hit_opened", 160.0, rank=4, auto=False, via="click"),
            self.ev("neighbour_opened", 100.5, rank=7),                     # A→B gap → B
            self.ev("neighbour_opened", 99.0, rank=9),                      # before A submit → not B
            self.ev("heatmap_toggled", 170.0, on=True, auto=True),
            self.ev("heatmap_toggled", 171.0, on=False),
            self.ev("search_mode_changed", 172.0, mode="roi"),
            self.ev("pecan_opened", 173.0),
            self.ev("llm_panel_toggled", 201.0, open=True, auto=True),
            self.ev("neighbour_opened", 200.5, rank=2),                     # B→C gap → neither
            self.ev("neighbour_opened", 250.0, rank=3),                     # during C
            self.ev("roi_selected", 301.0),                                 # after C submit
        ]
        v = d.engagement_vars(evs, **self.T)
        self.assertEqual(v["n_roi_selected"], 1)
        self.assertEqual(v["n_hit_opened"], 1)
        self.assertEqual(v["n_neighbour_opened"], 1)
        self.assertEqual(v["n_heatmap_toggled"], 1)
        self.assertEqual(v["n_search_mode_changed"], 1)
        self.assertEqual(v["n_pecan_opened"], 1)
        self.assertTrue(v["engaged_B"])
        self.assertEqual(v["deepest_rank_opened"], 7)
        self.assertEqual(v["n_retrieval_events_C"], 1)
        self.assertTrue(v["engaged_C_retrieval"])

    def test_list_only_reading(self):
        evs = [self.ev("roi_panel_toggled", 101.0, open=True, auto=True),
               self.ev("hit_opened", 150.0, rank=1, auto=True),
               self.ev("heatmap_toggled", 160.0, on=True),
               self.ev("search_mode_changed", 161.0, mode="roi"),
               self.ev("llm_panel_toggled", 250.0, open=False)]
        v = d.engagement_vars(evs, **self.T)
        self.assertFalse(v["engaged_B"])
        self.assertIsNone(v["deepest_rank_opened"])
        self.assertFalse(v["engaged_C_retrieval"])

    def test_no_c_submitted(self):
        v = d.engagement_vars([self.ev("neighbour_opened", 250.0, rank=1)], 100.0, 101.0, 200.0, None, None)
        self.assertFalse(v["engaged_C_retrieval"])
        self.assertEqual(d.phase_label(250.0, 100.0, 101.0, 200.0, None, None), "after_B")

    def test_phase_label(self):
        pl = lambda ts: d.phase_label(ts, **self.T)
        self.assertEqual([pl(50), pl(100.5), pl(200), pl(200.5), pl(201), pl(301)],
                         ["A", "B", "B", "gap_BC", "C", "after_C"])


class TestCaseCovariates(unittest.TestCase):
    def test_plurality_and_hits(self):
        labels = ["EWS", "RMS", "RMS", "NB", "RMS", "EWS", "EWS", "SS", "SS", "NB"]
        v = d.neighbour_retrieval(labels, "RMS")
        self.assertEqual((v["top1_correct"], v["hit5"], v["hit10"], v["n_truth_top10"]), (False, True, True, 3))
        self.assertFalse(v["majority10_correct"])        # RMS 3 vs EWS 3: tie → false
        self.assertEqual(v["plurality10_label"], "")
        labels2 = ["NB"] * 4 + ["RMS"] * 3 + ["EWS"] * 3
        v2 = d.neighbour_retrieval(labels2, "NB")
        self.assertTrue(v2["majority10_correct"])
        self.assertTrue(v2["top1_correct"])
        v3 = d.neighbour_retrieval(["EWS"] * 5 + ["WT"] * 5, "WT")
        self.assertEqual((v3["hit5"], v3["hit10"], v3["majority10_correct"]), (False, True, False))
        v4 = d.neighbour_retrieval(["RMS"] * 12, "RMS")
        self.assertEqual(v4["n_truth_top10"], 10)

    def test_roi_success(self):
        v = d.roi_success(["RMS", "EWS", None, "RMS"], "RMS")
        self.assertEqual(v, {"n_rois": 4, "n_rois_without_hits": 1, "roi_success_frac": 0.5})
        self.assertIsNone(d.roi_success([], "RMS")["roi_success_frac"])

    def test_library_support(self):
        self.assertEqual(d.library_support(["RMS", "RMS", "NB"]), {"RMS": 2, "NB": 1})

    def test_meta_visible(self):
        self.assertTrue(d.meta_visible_a(0.0, "CCDI"))
        self.assertFalse(d.meta_visible_a(d.META_VISIBLE_T, "STV"))
        self.assertTrue(d.meta_visible_a(d.META_VISIBLE_T + 1, "STV"))

    def test_chrono_and_sessions(self):
        self.assertEqual(d.chronological_rank([30.0, 10.0, 20.0]), [3, 1, 2])
        t = [0.0, 100.0, 1900.0, 3701.0, 50.0]
        # sorted: 0,50,100 | 1900 (gap 1800, not > 1800) | 3701 (gap 1801)
        self.assertEqual(d.session_ids(t), [1, 1, 1, 2, 1])

    def test_reader_exclusions(self):
        for rid in ("dev.one", "dev.two", "dev.three", "dev.four", "jane.doe", "test.user",
                    "demo.reader", "demo-af1a1583", "dev.five"):
            self.assertTrue(d.is_excluded_reader(rid), rid)
        self.assertFalse(d.is_excluded_reader("reader.one"))
        self.assertEqual(d.EXPERTISE_TIER["reader.two"], "expert")


if __name__ == "__main__":
    unittest.main()


class TestReadShaList(unittest.TestCase):
    """build_readings.read_sha_list: prod_hashes listings may carry a directory prefix (exports from
    2026-11-11 hashed as `sha256sum manifests/*.json`); SHA256SUMS keeps its relative paths."""

    def test_basename_normalises_prefix_but_keeps_digest(self):
        import build_readings as br
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "h.sha"
            p.write_text("aa11  manifests/X.json\nbb22  *Y.json\n")
            self.assertEqual(br.read_sha_list(p, basename=True), {"X.json": "aa11", "Y.json": "bb22"})
            self.assertEqual(br.read_sha_list(p), {"manifests/X.json": "aa11", "Y.json": "bb22"})
