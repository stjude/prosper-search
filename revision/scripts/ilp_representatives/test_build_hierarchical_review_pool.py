"""Synthetic-only fixtures for the [analysis plan] author-review selector."""
from __future__ import annotations

import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pulp import LpSolutionIntegerFeasible, LpSolutionOptimal
from openpyxl import load_workbook

import build_hierarchical_review_pool as v4


class HierarchicalPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(__file__).resolve().parent / f".v4_test_fixture_{os.getpid()}_{uuid.uuid4().hex}"
        cls.scratch.mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch)

    def fixture(self):
        ids = [f"S{i:02}" for i in range(42)]
        groups = ["RMS"] * 14 + ["NB"] * 13 + ["WT"] * 13 + ["RMS", "WT"]
        pop = pd.DataFrame({
            "slide_id": ids, "patient_id": [f"P{i}" for i in range(40)] + ["P0", "P41"],
            "tumor_group": groups, "comet_diagnosis": groups,
            "file_name": [f"{s}.tiff" for s in ids],
            "stv_dx": groups, "stv_subclass": [""] * 42,
        }).set_index("slide_id", drop=False)
        labels = (["other"] + ["difficult"] * 4 + ["very_difficult"] * 4 + ["moderate"] * 5 +
                  ["no_model_signal"] * 4 + ["other"] * 4)
        ali = pd.DataFrame({"slide_id": ids[:22] + [ids[-1]],
                            "selection_tier": ["Tier1"] * 22 + ["Tier2"],
                            "tier": labels + ["other"]}).set_index("slide_id")
        eligible = set(ids[:-1])
        anchors = ids[:8]
        rectangular = {s: anchors[:5] for s in ids[:-1]}
        for s in anchors:
            rectangular[s] = [s] + [a for a in anchors if a != s][:5]
        rectangular[ids[4]] = [ids[4], ids[5], ids[6], ids[7], ids[0], ids[1]]
        original = {a: ids for a in anchors}
        selected = ids[:40]
        tissue = [799, 800, 999, 1000] + [1200] * 38
        counts = pd.DataFrame({"detected_tissue_tiles": tissue,
                               "tissue_band": [v4.tissue_band(v) for v in tissue]}, index=ids)
        return pop, ali, eligible, anchors, original, rectangular, selected, counts

    def test_graph_direction_self_rectangular_ties_and_original_composition(self):
        ids = list("ABCDEFGH")
        matrix = pd.DataFrame(100.0, index=ids, columns=ids)
        np.fill_diagonal(matrix.values, 1_000_000)
        matrix.at["A", "D"] = matrix.at["A", "E"] = 0.1
        matrix.at["B", "F"] = matrix.at["B", "G"] = 0.1
        matrix.at["C", "H"] = matrix.at["C", "A"] = 0.1
        original, rectangular = v4.graphs(matrix, list("ABC"), ids, k=2)
        self.assertEqual(original["A"], ["A", "D", "E"])
        self.assertEqual(rectangular["A"], ["A", "B", "C"])
        self.assertEqual(rectangular["D"], ["A", "B"])
        self.assertEqual(len(rectangular["D"]), 2)
        self.assertEqual(v4.coverage(["A"], original, rectangular, list("ABC"))[0], set("ABC"))
        self.assertEqual(v4.coverage(["A"], original, rectangular, list("ABC"))[1], set(ids))
        rectangular["D"] = ["A"]
        self.assertNotEqual(v4.coverage(["D"], original, rectangular, list("ABC"))[1], set(ids))
        matrix.loc["C", ["H", "A"]] = 100
        with self.assertRaisesRegex(ValueError, "do not cover"):
            v4.graphs(matrix, list("ABC"), ids, k=2)
        ids6 = list("ABCDEF") + ["G"]
        six = pd.DataFrame(10.0, index=ids6, columns=ids6)
        np.fill_diagonal(six.values, 1_000_000)
        six.at["F", "G"] = 0.1
        _, six_rect = v4.graphs(six, ids6[:6], ids6, k=5)
        self.assertEqual(len(six_rect["A"]), 6)
        self.assertEqual(len(six_rect["G"]), 5)
        self.assertEqual(six_rect["G"], list("ABCDE"))

    def test_tissue_priority_exact_bands_and_integer(self):
        self.assertEqual([v4.tissue_band(i) for i in (799, 800, 999, 1000)],
                         ["low_lt800", "borderline_800_999", "borderline_800_999", "preferred_ge1000"])
        counts = pd.DataFrame({"detected_tissue_tiles": [799, 800, 999, 1000],
                               "mosaic_tiles": ["5", "3", "2", "1"]}, index=list("ABCD"))
        self.assertEqual(v4.tissue_cost(list("ABCD"), counts), 43)
        self.assertGreater(41, 40)
        for invalid in ("", "1.0", "-3", "NaN", " 2", "00"):
            counts.at["A", "mosaic_tiles"] = invalid
            with self.assertRaisesRegex(ValueError, "noninteger"):
                v4.integer_column(counts, "mosaic_tiles")

    def test_stv_provisional_ret_label_only_where_applicable(self):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.fixture()
        pop.at["S00", "stv_dx"] = "RET"
        rows = v4.case_table(selected, "stv", pop, ali, counts)
        self.assertIn("RET→RB provisional", rows.at[0, "label_qualification"])
        self.assertNotIn("RET→RB", rows.at[1, "label_qualification"])
        self.assertEqual(rows.qc_status.unique().tolist(), ["PENDING"])
        counts["cluster_duplicate_count"] = 0
        self.assertEqual(v4.case_table(selected, "stv", pop, ali, counts)
                         .cluster_duplicate_count.unique().tolist(), ["NOT_ASSESSED"])

    def test_ali_priority_and_unknown_difficulty(self):
        path = self.scratch / "ali.xlsx"
        with pd.ExcelWriter(path) as w:
            pd.DataFrame({"file_name": ["A.tiff", "B.tiff"],
                          "selection_tier": ["Tier1", "Tier2"],
                          "tier": ["no_signal", "easy"]}).to_excel(w, sheet_name="StJude", index=False)
        with patch.dict(v4.EXPECTED, {"stv": (3, 1, 2, 1)}):
            ali, eligible = v4.load_ali("stv", path, ["A", "B", "C"])
        self.assertEqual(eligible, {"A", "C"})
        self.assertEqual(ali.at["A", "tier"], "no_model_signal")
        for col, value in (("tier", "undefined"), ("selection_tier", "Tier3")):
            with pd.ExcelWriter(path) as w:
                pd.DataFrame({"file_name": ["A.tiff"], "selection_tier": ["Tier1"],
                              "tier": ["easy"]}).assign(**{col: value}).to_excel(
                                  w, sheet_name="StJude", index=False)
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                v4.load_ali("stv", path, ["A", "B", "C"])

    def test_selection_checks_quota_class_patient_and_tier2(self):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.fixture()
        valid = v4.selection_checks(selected, pop, ali, eligible, original, rectangular, anchors)
        self.assertTrue(all(item["passed"] for item in valid.values()))
        duplicate = selected[:-1] + ["S40"]
        checks = v4.selection_checks(duplicate, pop, ali, eligible, original, rectangular, anchors)
        self.assertFalse(checks["duplicate_patient"]["passed"])
        self.assertFalse(v4.selection_checks(selected[:-1] + ["S41"], pop, ali, eligible,
                                             original, {**rectangular, "S41": anchors[:5]},
                                             anchors)["ineligible_or_tier2"]["passed"])
        insufficient_hard = [s for s in selected if s != "S01"] + ["S40"]
        self.assertFalse(v4.selection_checks(insufficient_hard, pop, ali, eligible,
                                             original, rectangular, anchors)["ali_hard"]["passed"])
        many_rms = pop.copy()
        many_rms.at["S14", "tumor_group"] = "RMS"
        self.assertFalse(v4.selection_checks(selected, many_rms, ali, eligible, original,
                                             rectangular, anchors)["class:RMS"]["passed"])
        capped = pop.copy()
        for s in ("S00", "S01"):
            capped.at[s, "tumor_group"] = "RB"
        for s in ("S02", "S03", "S04"):
            capped.at[s, "tumor_group"] = "MEL"
        caps = v4.selection_checks(selected, capped, ali, eligible, original, rectangular, anchors)
        self.assertFalse(caps["class:RB"]["passed"])
        self.assertFalse(caps["class:MEL"]["passed"])
        ccdi = pop.drop(columns="stv_dx")
        ccdi_checks = v4.selection_checks(selected, ccdi, ali, eligible, original,
                                          rectangular, anchors)
        self.assertEqual(ccdi_checks["ali_no_model_signal"]["minimum"], 4)
        ccdi_ali = ali.copy()
        ccdi_ali.at["S14", "tier"] = "other"
        self.assertFalse(v4.selection_checks(selected, ccdi, ccdi_ali, eligible,
                                             original, rectangular, anchors)[
                                                 "ali_no_model_signal"]["passed"])

    def test_solver_proven_optimal_and_unproven_incumbent(self):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.fixture()
        configured = []
        actual_cbc = v4.PULP_CBC_CMD

        class ObservedCBC(actual_cbc):
            def actualSolve(self, model, **kwargs):
                configured.append((self.tmpDir, Path(self.tmpDir).is_dir(),
                                   self.optionsDict["threads"], self.timeLimit))
                return super().actualSolve(model, **kwargs)

        with patch.object(v4, "PULP_CBC_CMD", ObservedCBC):
            chosen, info = v4.solve(pop, ali, eligible, rectangular, anchors, counts,
                                    solver_dir=self.scratch)
        self.assertEqual(len(configured), 1)
        scratch_path, existed_during_solve, threads, seconds = configured[0]
        self.assertEqual(Path(scratch_path).parent, self.scratch)
        self.assertTrue(existed_during_solve)
        self.assertEqual((threads, seconds), (1, 300))
        self.assertFalse(Path(scratch_path).exists())
        self.assertEqual(len(chosen), 40)
        self.assertEqual(info["solution_status"], LpSolutionOptimal)
        self.assertTrue(all(v["passed"] for v in v4.selection_checks(
            chosen, pop, ali, eligible, original, rectangular, anchors).values()))

        class BrokenCBC(actual_cbc):
            def actualSolve(self, model, **kwargs):
                configured.append((self.tmpDir, Path(self.tmpDir).is_dir(), 1, 300))
                raise RuntimeError("simulated solver failure")

        with patch.object(v4, "PULP_CBC_CMD", BrokenCBC):
            with self.assertRaisesRegex(RuntimeError, "simulated solver failure"):
                v4.solve(pop, ali, eligible, rectangular, anchors, counts, solver_dir=self.scratch)
        self.assertTrue(configured[-1][1])
        self.assertFalse(Path(configured[-1][0]).exists())
        with self.assertRaisesRegex(ValueError, "repository-local"):
            v4.solve(pop, ali, eligible, rectangular, anchors, counts)

        class FakeSolver:
            def __init__(self, status, solution):
                self.status, self.solution = status, solution

            def actualSolve(self, model, **kwargs):
                model.status = self.status
                model.sol_status = self.solution
                return self.status

        for status, solution in ((-1, -1), (1, LpSolutionIntegerFeasible), (0, LpSolutionIntegerFeasible)):
            with self.assertRaisesRegex(ValueError, "No proven optimal"):
                v4.solve(pop, ali, eligible, rectangular, anchors, counts, FakeSolver(status, solution))
        impossible_ali = ali.iloc[:10].copy()
        with self.assertRaisesRegex(ValueError, "No proven optimal"):
            v4.solve(pop, impossible_ali, eligible, rectangular, anchors, counts,
                     solver_dir=self.scratch)

    def test_original_anchors_pinned_not_learned_from_changed_file(self):
        work = self.scratch / "stv_anchor_fixture"
        inventory = self.scratch / "anchor_inventory"
        work.mkdir()
        inventory.mkdir()
        pop, matrix = work / "population.csv", work / "distance_matrix.pkl"
        ali, counts = self.scratch / "bound_ali.xlsx", inventory / "stv_tissue_counts.csv"
        for file in (pop, matrix, ali, counts):
            file.write_text(file.name)
        anchor = work / "selected_round1.txt"
        anchor.write_text("A\n")
        (inventory / "COMPLETE").write_text("complete")
        (work / "input_sha256.json").write_text(json.dumps({
            str(pop): v4.digest(pop), str(matrix): v4.digest(matrix),
        }))
        manifest = inventory / "source_sha256.json"
        manifest.write_text(json.dumps({str(p): v4.digest(p) for p in (pop, matrix, ali, counts)}))
        with (patch.object(v4, "SOURCE_MANIFEST_SHA", v4.digest(manifest)),
              patch.object(v4, "ALI_FILE", ali),
              patch.dict(v4.COUNT_SHA, {"stv": v4.digest(counts)}),
              patch.dict(v4.ANCHOR_SHA, {"stv": v4.digest(anchor)})):
            bound = v4.source_bindings("stv", work, inventory)
            self.assertEqual(bound[str(anchor)], v4.digest(anchor))
            anchor.write_text("B\n")
            with self.assertRaisesRegex(ValueError, "Source binding changed"):
                v4.source_bindings("stv", work, inventory)

    def test_single_swap_valid_alone_and_invalid_outgoing(self):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.fixture()
        options = v4.single_swaps(selected, pop, ali, eligible, original, rectangular, anchors, counts)
        self.assertEqual(len(options), 1)
        self.assertEqual((options[0]["incoming_slide_id"], options[0]["outgoing_slide_id"]),
                         ("S40", "S00"))
        self.assertEqual(options[0]["tissue_cost_delta"], -41)
        self.assertEqual(options[0]["incoming_qc_flag"], "ORDINARY_IMAGE_QC_REQUIRED")
        self.assertEqual(v4.single_swaps(selected, pop, ali, set(selected),
                                          original, rectangular, anchors, counts), [])
        self.assertFalse(v4.selection_checks(selected[1:] + ["S41"], pop, ali, eligible,
                                             original, {**rectangular, "S41": anchors[:5]},
                                             anchors)["ineligible_or_tier2"]["passed"])

    def test_count_manifest_integers_duplicate_missing_and_lineage(self):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.fixture()
        path = self.scratch / "counts.csv"
        source = self.scratch / "source.bin"
        source.write_bytes(b"source")
        sha = v4.digest(source)
        rows = pd.DataFrame({
            "slide_id": ["A", "B"], "detected_tissue_tiles": ["799", "1000"],
            "mosaic_tiles": ["40", "120"], "tissue_band": ["low_lt800", "preferred_ge1000"],
            "count_method": ["original_stdout", "original_provenance"],
            "count_source_path": [str(source)] * 2, "count_source_sha256": [sha] * 2,
            "mosaic_meta_path": [str(source)] * 2, "mosaic_meta_sha256": [sha] * 2,
            "tile_size_px": ["512"] * 2, "tissue_mask_threshold": ["0.7"] * 2,
            "cluster_duplicate_count": ["0"] * 2,
        })

        def check(df, expect_error=None):
            df.to_csv(path, index=False)
            with patch.dict(v4.COUNT_SHA, {"stv": v4.digest(path)}):
                if expect_error:
                    with self.assertRaisesRegex(ValueError, expect_error):
                        v4.load_counts("stv", path, ["A", "B"], {str(source): sha})
                else:
                    loaded = v4.load_counts("stv", path, ["A", "B"], {str(source): sha})
                    self.assertEqual(loaded.detected_tissue_tiles.tolist(), [799, 1000])
                    self.assertEqual(loaded.cluster_duplicate_count.tolist(),
                                     ["NOT_ASSESSED", "NOT_ASSESSED"])
        check(rows)
        check(rows.iloc[[0, 0]], "duplicate")
        check(rows.iloc[:1], "exactly")
        bad = rows.copy()
        bad.at[0, "detected_tissue_tiles"] = "799.0"
        check(bad, "noninteger")
        bad = rows.copy()
        bad.at[0, "count_source_sha256"] = "0" * 64
        check(bad, "Unbound")
        bad = rows.copy()
        bad.at[0, "tissue_band"] = "preferred_ge1000"
        check(bad, "bands")
        ccdi = rows.copy()
        ccdi["count_method"] = "feature_and_mosaic_unique_cluster_locs"
        ccdi.at[0, "cluster_duplicate_count"] = "1"
        ccdi.to_csv(path, index=False)
        with patch.dict(v4.COUNT_SHA, {"ccdi": v4.digest(path)}):
            with self.assertRaisesRegex(ValueError, "duplicate-free"):
                v4.load_counts("ccdi", path, ["A", "B"], {str(source): sha})

    def test_matrix_identity_and_distance_guard(self):
        ids = ["A", "B"]
        matrix = pd.DataFrame([[1e6, 0.3], [0.3, 1e6]], index=ids, columns=ids)
        path = self.scratch / "matrix.pkl"
        matrix.to_pickle(path)
        v4.load_distance(path, ids)
        with self.assertRaisesRegex(ValueError, "ordered"):
            v4.load_distance(path, ids[::-1])
        matrix.iat[0, 1] = np.nan
        matrix.to_pickle(path)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            v4.load_distance(path, ids)

    def test_outputs_readback_protection_and_binding_mutation(self):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.fixture()
        counts = counts.copy()
        source = self.scratch / "binding.txt"
        source.write_text("stable")
        bindings = {str(source): v4.digest(source)}
        for col in ("mosaic_tiles", "cluster_duplicate_count"):
            counts[col] = [1] * len(counts)
        counts["count_method"] = "original_stdout"
        counts["count_source_path"] = str(source)
        counts["count_source_sha256"] = bindings[str(source)]
        counts["mosaic_meta_path"] = str(source)
        counts["mosaic_meta_sha256"] = bindings[str(source)]
        counts["tile_size_px"] = 512
        counts["tissue_mask_threshold"] = "0.7"
        run = self.scratch / "run"
        workbook = self.scratch / "review.xlsx"
        v4.write_outputs(run, workbook, "stv", selected, pop, ali, eligible, counts,
                         original, rectangular, anchors, {"problem_status": "Optimal",
                                                            "solution_status": LpSolutionOptimal,
                                                            "objective": 43}, bindings)
        self.assertTrue((run / "COMPLETE").is_file())
        self.assertEqual(pd.read_excel(workbook, sheet_name="pool").slide_id.tolist(), selected)
        self.assertEqual(pd.read_csv(run / "pool.csv").slide_id.tolist(), selected)
        self.assertEqual(pd.read_excel(workbook, sheet_name="pool")
                         .cluster_duplicate_count.unique().tolist(), ["NOT_ASSESSED"])
        self.assertEqual(json.loads((run / "run.json").read_text())["single_swap_option_count"], 1)
        receipt = json.loads((run / "COMPLETE").read_text())
        self.assertEqual(receipt["status"], "SOURCE_BOUND_AUTHOR_QC_REVIEW_ONLY")
        for file, sha in receipt["output_sha256"].items():
            self.assertEqual(v4.digest(Path(file)), sha)
        with self.assertRaisesRegex(FileExistsError, "Existing"):
            v4.write_outputs(run, workbook, "stv", selected, pop, ali, eligible, counts,
                             original, rectangular, anchors, {}, bindings)
        run_empty = self.scratch / "run_empty"
        workbook_empty = self.scratch / "review_empty.xlsx"
        v4.write_outputs(run_empty, workbook_empty, "stv", selected, pop, ali, set(selected),
                         counts, original, rectangular, anchors, {}, bindings)
        self.assertEqual(json.loads((run_empty / "run.json").read_text())["single_swap_option_count"], 0)
        self.assertIn("incoming_slide_id",
                      pd.read_excel(workbook_empty, sheet_name="single_swaps").columns)
        run_blocked = self.scratch / "run_blocked"
        with self.assertRaisesRegex(FileExistsError, "Existing"):
            v4.write_outputs(run_blocked, workbook, "stv", selected, pop, ali, eligible, counts,
                             original, rectangular, anchors, {}, bindings)
        self.assertFalse(run_blocked.exists())
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            v4.verify_bindings(bindings)
        run2 = self.scratch / "run2"
        workbook2 = self.scratch / "review2.xlsx"
        with self.assertRaisesRegex(ValueError, "binding changed"):
            v4.write_outputs(run2, workbook2, "stv", selected, pop, ali, eligible, counts,
                             original, rectangular, anchors, {}, bindings)
        self.assertFalse((run2 / "COMPLETE").exists())

    def test_late_workbook_report_and_swap_mutations_leave_no_receipt(self):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.fixture()
        source = self.scratch / "late_source.txt"
        source.write_text("stable")
        bindings = {str(source): v4.digest(source)}
        original_verify = v4.verify_bindings
        for name in ("workbook", "report", "swaps"):
            with self.subTest(name=name):
                run = self.scratch / f"late_{name}"
                workbook = self.scratch / f"late_{name}.xlsx"

                def mutate_on_source_pass(current):
                    original_verify(current)
                    if current is not bindings:
                        return
                    if name == "workbook":
                        book = load_workbook(workbook)
                        book["pool"]["A2"] = "FABRICATED_QC_PASS"
                        book.save(workbook)
                    elif name == "report":
                        report_path = run / "run.json"
                        report = json.loads(report_path.read_text())
                        report["objective"]["value"] = -1
                        report_path.write_text(json.dumps(report))
                    else:
                        swaps = run / "single_swaps.csv"
                        table = pd.read_csv(swaps)
                        table.loc[0, "incoming_qc_flag"] = "FABRICATED_QC_PASS"
                        table.to_csv(swaps, index=False)

                with patch.object(v4, "verify_bindings", side_effect=mutate_on_source_pass):
                    with self.assertRaisesRegex(ValueError, "Source binding changed"):
                        v4.write_outputs(run, workbook, "stv", selected, pop, ali, eligible,
                                         counts, original, rectangular, anchors, {}, bindings)
                self.assertFalse((run / "COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
