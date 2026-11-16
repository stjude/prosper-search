"""End-to-end smoke test of analyze.py on fully synthetic inputs (no study data)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze  # noqa: E402
import derive as d  # noqa: E402

GROUPS = ["RMS", "NB", "WT", "EWS", "OS"]


def synthetic(root: Path, n_cases: int = 12):
    rng = np.random.default_rng(0)
    analysis, export, data = root / "analysis", root / "export", root / "data"
    for p in (analysis, export / "study_config", data / "manifests", data / "llm"):
        p.mkdir(parents=True)
    lib = [{"slide_id": f"L{i}", "kind": "library", "label": GROUPS[i % 5]} for i in range(40)]
    pd.DataFrame(lib).to_csv(export / "study_config" / "slides_index.csv", index=False)
    rows, subs = [], []
    for rid in ("reader.one", "reader.two"):
        for i in range(n_cases):
            cid, sid, truth = f"case{i + 10:03d}", f"S{i}", GROUPS[i % 5]
            row = {"reader_id": rid, "expertise_tier": d.EXPERTISE_TIER[rid], "case_id": cid, "slide_id": sid,
                   "cohort": "CCDI" if i < n_cases // 2 else "STV", "reference": truth, "has_C": True,
                   "stv_subclass": "" if i < n_cases // 2 else "embryonal", "ccdi_diagnosis": "x",
                   "changed_B": rng.choice(["yes", "no"]), "changed_C": rng.choice(["yes", "no"])}
            for p in "ABC":
                fav = bool(rng.random() < 0.5)
                ind = fav or bool(rng.random() < 0.5)
                favd = truth if fav else (d.MISC if rng.random() < 0.2 else GROUPS[(i + 1) % 5])
                dx = [favd] + ([truth] if ind and not fav else [])
                row.update({f"dx_{p}": " | ".join(dx), f"favoured_{p}": favd, f"eff_source_{p}": p,
                            f"n_dx_{p}": len(dx), f"fav_correct_{p}": fav, f"in_diff_{p}": ind,
                            f"truth_rank_{p}": (1.0 if fav else 2.0) if ind else np.nan,
                            f"misc_in_{p}": d.MISC in dx, f"normal_in_{p}": False,
                            f"cmap_note_{p}": ("misc_mapped:x->" + truth) if d.MISC in dx else "",
                            f"fav_correct_{p}_cmap": fav or d.MISC == favd, f"in_diff_{p}_cmap": ind or d.MISC in dx,
                            f"conf_{p}": int(rng.integers(1, 6)), f"sec_{p}": float(rng.integers(20, 400)),
                            f"interrupted_{p}": False,
                            f"comments_{p}": f"comment {p}\nembryonal sarcoma, alveolar" if i == 0 else f"comment {p}\nline two"})
            for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
                changed = row["changed_B"] == "yes" if (x, y) == ("A", "B") else row["changed_C"] == "yes"
                c, s = d.outcome_class(row[f"fav_correct_{x}"], row[f"in_diff_{x}"], row[f"fav_correct_{y}"],
                                       row[f"in_diff_{y}"], changed)
                row[f"class_{x}{y}"], row[f"confirm_sub_{x}{y}"] = c, s
            row.update({"use_B": int(rng.integers(1, 6)), "use_C": int(rng.integers(1, 6)),
                        "dconf_BA": row["conf_B"] - row["conf_A"], "dconf_CB": row["conf_C"] - row["conf_B"],
                        "harmful_conf_B": row["conf_B"] > row["conf_A"] and not row["fav_correct_B"],
                        "interrupted_between_phases": i == 3, "gap_AB_s": 5.0, "gap_BC_s": 5.0,
                        "n_roi_selected": int(rng.integers(0, 2)), "n_hit_opened": 0, "n_neighbour_opened": int(i == 2),
                        "n_heatmap_toggled": 0, "n_search_mode_changed": 0, "n_pecan_opened": 0,
                        "engaged_B": i in (1, 2), "deepest_rank_opened": 3.0 if i == 2 else np.nan,
                        "n_retrieval_events_C": 0, "engaged_C_retrieval": False,
                        "top1_correct": i % 3 == 0, "hit5": i % 4 != 0, "hit10": True, "majority10_correct": i % 2 == 0,
                        "plurality10_label": truth, "library_support_n": 8,
                        "difficulty_tier": "Tier1" if i % 2 else "not_on_ali_list", "meta_visible_A": False,
                        "chrono_rank": i + 1, "session_id": i // 6,
                        "llm_top": truth if i % 3 else GROUPS[(i + 2) % 5], "llm_top_correct": bool(i % 3),
                        "llm_names_truth_leading": bool(i % 3), "llm_flags_discrepancy": True,
                        "llm_wording_probability": False})
            rows.append(row)
            for p in "ABC":
                subs.append({"reader_id": rid, "case_id": cid, "phase": p})
        for p in "AB":
            subs.append({"reader_id": rid, "case_id": "case01", "phase": p})
    pd.DataFrame(rows).to_csv(analysis / "readings.csv", index=False)
    ev = [{"reader_id": "reader.one", "case_id": "case010", "kind": "llm_panel_toggled", "auto": a}
          for a in (True, False, False)]
    pd.DataFrame(ev).to_csv(analysis / "events_engagement.csv", index=False)
    pd.DataFrame(subs).to_csv(export / "submissions_export.csv", index=False)
    for i in range(n_cases):
        nb = [{"rank": k + 1, "label": GROUPS[(i + k) % 5]} for k in range(10)]
        (data / "manifests" / f"S{i}.json").write_text(json.dumps({"slide_neighbours": nb}))
        (data / "llm" / f"S{i}.txt").write_text("No discrepant neighbors." if i % 2 else "Votes are split.")
    sha = hashlib.sha256((analysis / "readings.csv").read_bytes()).hexdigest()
    (analysis / "RUN_RECORD.generated.md").write_text(f"- output sha256:\n  - readings.csv: {sha}\n")
    return analysis, export, data


class TestAnalyzeSmoke(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis, export, data = synthetic(Path(tmp))
            rc = analyze.main(["--analysis", str(analysis), "--export", str(export), "--data", str(data),
                               "--readers", "reader.one,reader.two"])
            self.assertEqual(rc, 0)
            t2 = pd.read_csv(analysis / "tables" / "T2_primary_endpoints.csv")
            r = pd.read_csv(analysis / "readings.csv")
            fm = r[r.reader_id == "reader.one"]
            row = t2[(t2.reader_id == "reader.one") & (t2.endpoint == "E2 fav_correct_B")].iloc[0]
            self.assertEqual(row.k, fm.fav_correct_B.sum())
            self.assertIn("pooled", set(t2.reader_id))
            t9 = pd.read_csv(analysis / "tables" / "T9_comments_for_coding.csv", keep_default_na=False)
            self.assertEqual(len(t9), 2 * 12 * 3)
            self.assertTrue(all(c in t9.columns for c in analyze.CODES))
            t1 = pd.read_csv(analysis / "tables" / "T1_reader_case_flow.csv")
            self.assertEqual(t1.pilot_submissions_excluded.tolist(), [2, 2])
            for f in ("F1_trajectory_reader.one", "F4_retrieval_vs_class_reader.two"):
                for ext in ("png", "pdf"):
                    self.assertTrue((analysis / "figures" / f"{f}.{ext}").stat().st_size > 0)
            caps = (analysis / "tables" / "CAPTIONS.md").read_text()
            self.assertIn("C follows B; the summary's effect is not isolated", caps)
            self.assertTrue((analysis / "OUTPUTS_SHA256").exists())
            t5 = pd.read_csv(analysis / "tables" / "T5a_engagement_5_4.csv", keep_default_na=False)
            tot = t5[(t5.reader_id == "reader.one") & (t5.measure == "n_llm_toggled (total events)")]
            self.assertEqual(int(tot["count"].iloc[0]), 2)
            s1 = pd.read_csv(analysis / "tables" / "S1_trajectory_5_1.csv")
            self.assertEqual(set(s1.engaged_B_subset), {"all", "engaged_B = True", "engaged_B = False"})
            s4b = pd.read_csv(analysis / "tables" / "S4b_comment_named_entities_5_8.csv")
            self.assertEqual(set(s4b.map_outcome), {"LIVER"})
            s6 = pd.read_csv(analysis / "tables" / "S6_subtype_mentions_for_coding_5_8.csv", keep_default_na=False)
            self.assertIn("alveolar; embryonal", set(s6.plan_subtype_terms_found))
            t10 = pd.read_csv(analysis / "tables" / "T10_between_reader_5_7.csv")
            self.assertEqual(list(t10.step), ["A", "B", "C"])
            self.assertEqual(set(t10.paired_set_n), {12})
            for _, x in t10.iterrows():
                self.assertEqual(x.both_correct + x.only_reader1_correct + x.only_reader2_correct
                                 + x.neither_correct, x.n)
            t10b = pd.read_csv(analysis / "tables" / "T10b_paired_cases_5_7.csv")
            self.assertEqual(len(t10b), 12)
            self.assertIn("paired set n = reader.one vs reader.two: n = 12", caps)

    def test_single_reader_has_no_pooled_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis, export, data = synthetic(Path(tmp))
            analyze.main(["--analysis", str(analysis), "--export", str(export), "--data", str(data),
                          "--readers", "reader.one"])
            t2 = pd.read_csv(analysis / "tables" / "T2_primary_endpoints.csv")
            self.assertEqual(set(t2.reader_id), {"reader.one"})
            self.assertFalse((analysis / "tables" / "T10_between_reader_5_7.csv").exists())
            self.assertNotIn("T10", (analysis / "tables" / "CAPTIONS.md").read_text())

    def test_rejects_excluded_reader(self):
        with self.assertRaises(SystemExit):
            analyze.main(["--readers", "dev.four"])


if __name__ == "__main__":
    unittest.main()
