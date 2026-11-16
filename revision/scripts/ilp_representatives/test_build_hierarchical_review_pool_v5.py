"""Synthetic-only tests of the [analysis plan] uniformly filtered two-stage pool."""
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
from openpyxl import load_workbook
from pulp import LpProblem, LpMinimize, LpSolutionIntegerFeasible, LpSolutionOptimal

import build_hierarchical_review_pool_v5 as v5


class V5Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(__file__).resolve().parent / f".v5_test_fixture_{os.getpid()}_{uuid.uuid4().hex}"
        cls.scratch.mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch)

    def source(self, name):
        path = self.scratch / name
        path.write_text("synthetic source")
        return {str(path): v5.v4.digest(path)}

    def small(self):
        ids = list("ABCDEFGX")
        groups = ["RMS", "RMS", "RMS", "WT", "WT", "WT", "RMS", "WT"]
        base = pd.DataFrame({
            "slide_id": ids, "file_name": [f"{s}.tiff" for s in ids],
            "patient_id": [f"P{s}" for s in ids], "tumor_group": groups,
            "stv_dx": groups, "stv_subclass": [""] * len(ids), "comet_diagnosis": groups,
        }).set_index("slide_id", drop=False)
        matrix = pd.DataFrame(10.0, index=ids, columns=ids)
        np.fill_diagonal(matrix.values, 1_000_000)
        matrix.at["A", "X"] = 0.01
        ali = pd.DataFrame({"slide_id": ["A", "B", "X"],
                            "selection_tier": ["Tier1", "Tier2", "Tier1"],
                            "tier": ["moderate", "difficult", "very_difficult"]}).set_index("slide_id")
        counts = pd.DataFrame({"detected_tissue_tiles": [800, 900, 1000, 1000, 1000, 1000, 1000, 799],
                               "mosaic_tiles": [50] * 8, "cluster_duplicate_count": ["NOT_ASSESSED"] * 8},
                              index=ids)
        counts["tissue_band"] = counts.detected_tissue_tiles.map(v5.v4.tissue_band)
        return base, matrix, ali, set(ids) - {"B"}, counts

    def large(self):
        ids = [f"S{i:02}" for i in range(42)]
        groups = ["RMS"] * 14 + ["NB"] * 13 + ["WT"] * 13 + ["RMS", "WT"]
        pop = pd.DataFrame({
            "slide_id": ids, "file_name": [f"{s}.tiff" for s in ids],
            "patient_id": [f"P{i}" for i in range(40)] + ["P0", "P41"],
            "tumor_group": groups, "comet_diagnosis": groups,
            "stv_dx": groups, "stv_subclass": [""] * len(ids),
        }).set_index("slide_id", drop=False)
        labels = (["other"] + ["difficult"] * 4 + ["very_difficult"] * 4 +
                  ["moderate"] * 5 + ["no_model_signal"] * 4 + ["other"] * 4)
        ali = pd.DataFrame({"slide_id": ids[:22] + [ids[-1]],
                            "selection_tier": ["Tier1"] * 22 + ["Tier2"],
                            "tier": labels + ["other"]}).set_index("slide_id")
        eligible = set(ids[:-1])
        anchors = ids[:8]
        rectangular = {s: anchors[:5] for s in ids[:-1]}
        for s in anchors:
            rectangular[s] = [s] + [a for a in anchors if a != s][:5]
        rectangular["S04"] = ["S04", "S05", "S06", "S07", "S00", "S01"]
        original = {a: ids for a in anchors}
        counts = pd.DataFrame({"detected_tissue_tiles": [800] + [1000] * 41,
                               "mosaic_tiles": [50] * 42,
                               "cluster_duplicate_count": ["NOT_ASSESSED"] * 42},
                              index=ids)
        counts["tissue_band"] = counts.detected_tissue_tiles.map(v5.v4.tissue_band)
        return pop, ali, eligible, anchors, original, rectangular, ids[:40], counts

    def test_filter_applies_before_both_stages_to_targets_ali_and_nonali(self):
        base, matrix, ali, candidates, counts = self.small()
        with patch.dict(v5.EXPECTED, {"stv": (7, 6, 1, 2)}):
            pop, filtered, retained, eligible, manifest = v5.study_subset(
                "stv", base, matrix, ali, candidates, counts)
        self.assertEqual(list(pop.index), list("ABCDEFG"))
        self.assertEqual(eligible, set("ACDEFG"))
        self.assertIn("B", pop.index)  # Tier2 is a target/possible first-stage anchor.
        self.assertNotIn("X", pop.index)  # a co-author Tier1 with 799 cannot be a target.
        self.assertEqual(manifest.slide_id.tolist(), list(base.index))
        self.assertEqual(manifest.loc[manifest.slide_id == "X", "eligibility_reason"].item(),
                         "BELOW_800_TISSUE_TILES")
        self.assertEqual(manifest.loc[manifest.slide_id == "B", "eligibility_reason"].item(),
                         "ALI_TIER2_NOT_SELECTABLE")
        self.assertTrue(manifest.loc[manifest.slide_id == "A", "final_candidate"].item())
        self.assertEqual(filtered.shape, (7, 7))
        self.assertEqual(retained.at["A", "detected_tissue_tiles"], 800)
        neighbors = v5.neighborhoods(filtered)
        self.assertEqual(neighbors["A"], ["A", "B", "C", "D", "E", "F"])
        self.assertNotIn("X", neighbors["A"])
        self.assertIn("F", neighbors["A"])  # Newly fifth OTHER neighbor; never trim old lists.
        with self.assertRaisesRegex(ValueError, "Unexpected study population"):
            v5.study_subset("stv", base, matrix, ali, candidates - {"A"}, counts)

    def test_new_stage1_minimum_outgoing_cover_and_both_statuses(self):
        base, matrix, ali, candidates, counts = self.small()
        nb = v5.neighborhoods(matrix.loc[list("ABCDEFG"), list("ABCDEFG")])
        anchors, witness, info = v5.solve_stage1(nb, self.scratch)
        self.assertEqual(info["objective"], 2)
        self.assertEqual(len(anchors), 2)
        self.assertTrue(all(witness[s] for s in nb))
        self.assertEqual(set(witness), set("ABCDEFG"))
        self.assertTrue(all(s in nb[a] for s, covering in witness.items() for a in covering))
        self.assertNotIn("X", anchors)
        with self.assertRaisesRegex(ValueError, "outgoing"):
            v5.solve_stage1({**nb, "A": ["B", "A", "C", "D", "E", "F"]}, self.scratch)

        class FakeSolver:
            def __init__(self, status, solution):
                self.status, self.solution = status, solution

            def actualSolve(self, model, **kwargs):
                model.status, model.sol_status = self.status, self.solution
                return self.status

        for status, solution in ((-1, -1), (1, LpSolutionIntegerFeasible), (0, LpSolutionIntegerFeasible)):
            with self.subTest(status=status, solution=solution):
                with self.assertRaisesRegex(ValueError, "not proven optimal"):
                    v5.solve_stage1(nb, self.scratch, FakeSolver(status, solution))
        self.assertNotEqual(anchors, ["A", "B", "C", "D", "E", "F", "G"])

    def test_cbc_owned_scratch_both_stages_cleanup_and_failure(self):
        observed = []
        actual = v5.PULP_CBC_CMD

        class CheckedCBC(actual):
            def actualSolve(self, model, **kwargs):
                observed.append((self.tmpDir, Path(self.tmpDir).is_dir(),
                                 self.optionsDict["threads"], self.timeLimit))
                return super().actualSolve(model, **kwargs)

        nb = v5.neighborhoods(self.small()[1].loc[list("ABCDEFG"), list("ABCDEFG")])
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.large()
        with patch.object(v5, "PULP_CBC_CMD", CheckedCBC):
            v5.solve_stage1(nb, self.scratch)
            v5.solve_stage2(pop, ali, counts, eligible, anchors, original, rectangular, self.scratch)
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(Path(path).parent == self.scratch and existed and
                            threads == 1 and limit == 300 and not Path(path).exists()
                            for path, existed, threads, limit in observed))

        class FailedCBC(actual):
            def actualSolve(self, model, **kwargs):
                observed.append((self.tmpDir, Path(self.tmpDir).is_dir(), 1, 300))
                raise RuntimeError("solver failed")

        with patch.object(v5, "PULP_CBC_CMD", FailedCBC):
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                v5.solve_stage1(nb, self.scratch)
        self.assertTrue(observed[-1][1])
        self.assertFalse(Path(observed[-1][0]).exists())

    def test_zero_objective_and_quota_patient_guards(self):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.large()
        minimum, _, first_info = v5.solve_stage2(
            pop, ali, counts, eligible, anchors, original, rectangular, self.scratch)
        self.assertEqual(first_info["objective"], 0)
        self.assertNotIn("S00", minimum)  # Borderline cost is avoided when a valid preferred swap exists.
        counts["detected_tissue_tiles"] = 1000
        counts["tissue_band"] = "preferred_ge1000"
        chosen, checks, info = v5.solve_stage2(
            pop, ali, counts, eligible, anchors, original, rectangular, self.scratch)
        self.assertEqual(info["objective"], 0)
        self.assertTrue(all(row["passed"] for row in checks.values()))
        self.assertEqual(len({pop.at[s, "patient_id"] for s in chosen}), 40)
        self.assertEqual(len(chosen), 40)
        tier1 = ali.iloc[:19].copy()
        with self.assertRaisesRegex(ValueError, "not proven optimal"):
            v5.solve_stage2(pop, tier1, counts, eligible, anchors, original,
                            rectangular, self.scratch)
        counts.at["S00", "detected_tissue_tiles"] = 799
        with self.assertRaisesRegex(ValueError, "ineligible tissue"):
            v5.solve_stage2(pop, ali, counts, eligible, anchors, original,
                            rectangular, self.scratch)
        counts.at["S00", "detected_tissue_tiles"] = 1000
        ccdi = pop.drop(columns="stv_dx")
        missing_signal = ali.copy()
        missing_signal.at["S14", "tier"] = "other"
        with self.assertRaisesRegex(ValueError, "not proven optimal"):
            v5.solve_stage2(ccdi, missing_signal, counts, eligible, anchors, original,
                            rectangular, self.scratch)
        without_wt = {s for s in eligible if pop.at[s, "tumor_group"] != "WT"}
        with self.assertRaisesRegex(ValueError, "no eligible final candidate for group WT"):
            v5.solve_stage2(pop, ali, counts, without_wt, anchors, original,
                            rectangular, self.scratch)

    def test_failed_second_stage_preserves_bound_new_stage1_only(self):
        base, matrix, ali, candidates, counts = self.small()
        run = self.scratch / "only_stage1"
        workbook = self.scratch / "no_final.xlsx"
        source = self.source("stage1_source.txt")
        with patch.dict(v5.EXPECTED, {"stv": (7, 6, 1, 2)}):
            with self.assertRaisesRegex(ValueError, "Insufficient anchors"):
                v5.execute("stv", run, workbook, base, matrix, ali, candidates,
                           counts, source, self.scratch)
        receipt = json.loads((run / "STAGE1_COMPLETE").read_text())
        self.assertEqual(receipt["status"], "STAGE1_OPTIMAL_ONLY_NO_FINAL_POOL")
        self.assertFalse((run / "COMPLETE").exists())
        self.assertFalse(workbook.exists())
        stage1 = json.loads((run / "stage1.json").read_text())
        self.assertEqual(stage1["selected_anchor_ids"],
                         (run / "stage1_anchors.txt").read_text().splitlines())
        self.assertEqual(stage1["anchor_list_sha256"], v5.v4.digest(run / "stage1_anchors.txt"))
        for file, sha in receipt["artifact_sha256"].items():
            self.assertEqual(v5.v4.digest(Path(file)), sha)
        self.assertEqual(pd.read_pickle(run / "filtered_distance_matrix.pkl").shape, (7, 7))
        with self.assertRaisesRegex(FileExistsError, "Existing"):
            v5.execute("stv", run, workbook, base, matrix, ali, candidates,
                       counts, source, self.scratch)

    def final_fixture(self, stem):
        pop, ali, eligible, anchors, original, rectangular, selected, counts = self.large()
        source = self.source(f"{stem}_source.txt")
        run = self.scratch / stem
        run.mkdir()
        for name in ("filtered_population.csv", "filtered_distance_matrix.pkl",
                     "filtered_tissue_counts.csv", "base_eligibility.csv"):
            (run / name).write_text("synthetic stage1 artifact")
        anchor_file = run / "stage1_anchors.txt"
        anchor_file.write_text("\n".join(anchors) + "\n")
        (run / "stage1.json").write_text(json.dumps({
            "selected_anchor_ids": anchors, "anchor_list_sha256": v5.v4.digest(anchor_file),
        }))
        artifacts = v5.file_bindings([run / name for name in (
            "filtered_population.csv", "filtered_distance_matrix.pkl",
            "filtered_tissue_counts.csv", "base_eligibility.csv",
            "stage1_anchors.txt", "stage1.json")])
        marker = run / "STAGE1_COMPLETE"
        marker.write_text(json.dumps({
            "status": "STAGE1_OPTIMAL_ONLY_NO_FINAL_POOL",
            "source_binding_set_sha256": v5.hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()).hexdigest(),
            "artifact_sha256": artifacts,
        }))
        staged = {**artifacts, str(marker): v5.v4.digest(marker)}
        eligibility = pd.DataFrame({"slide_id": list(pop.index),
                                    "quantity_eligible": [True] * len(pop),
                                    "final_candidate": [s in eligible for s in pop.index]})
        checks = v5.v4.selection_checks(selected, pop, ali, eligible, original, rectangular, anchors)
        return (pop, ali, eligible, anchors, original, rectangular, selected,
                counts, source, staged, eligibility, checks)

    def write_final_fixture(self, stem, mutation=None):
        (pop, ali, eligible, anchors, original, rectangular, selected, counts,
         source, staged, eligibility, checks) = self.final_fixture(stem)
        run = self.scratch / stem
        workbook = self.scratch / f"{stem}.xlsx"
        original_verify = v5.v4.verify_bindings

        def mutate_at_source_pass(current):
            original_verify(current)
            if current is not source or mutation is None:
                return
            if mutation == "workbook":
                book = load_workbook(workbook)
                book["pool"]["A2"] = "FABRICATED_QC_PASS"
                book.save(workbook)
            elif mutation == "report":
                file = run / "run.json"
                changed = json.loads(file.read_text())
                changed["stage2_solver"]["objective"] = -1
                file.write_text(json.dumps(changed))
            elif mutation == "swaps":
                file = run / "single_swaps.csv"
                data = pd.read_csv(file)
                data.loc[0, "incoming_qc_flag"] = "FABRICATED_QC_PASS"
                data.to_csv(file, index=False)
            elif mutation == "anchor":
                (run / "stage1_anchors.txt").write_text("swapped anchor set")
            elif mutation == "source":
                Path(next(iter(source))).write_text("changed during final verification")

        with patch.object(v5.v4, "verify_bindings", side_effect=mutate_at_source_pass):
            v5.final_write(run, workbook, "stv", len(pop), pop, ali, counts, eligible,
                           anchors, original, rectangular, selected, checks,
                           {"objective": 8}, {"objective": 1},
                           eligibility, source, staged)
        return run, workbook

    def test_final_receipt_all_artifacts_pending_qc_no_below800_swaps(self):
        run, workbook = self.write_final_fixture("final_success")
        marker = json.loads((run / "COMPLETE").read_text())
        self.assertEqual(marker["status"], "V5_SOURCE_BOUND_AUTHOR_QC_REVIEW_ONLY")
        for path, sha in marker["output_sha256"].items():
            self.assertEqual(v5.v4.digest(Path(path)), sha)
        self.assertIn(str(workbook), marker["output_sha256"])
        pool = pd.read_csv(run / "pool.csv")
        self.assertEqual(pool.qc_status.unique().tolist(), ["PENDING"])
        self.assertGreaterEqual(pool.detected_tissue_tiles.min(), 800)
        self.assertEqual(pool.cluster_duplicate_count.unique().tolist(), ["NOT_ASSESSED"])
        swaps = pd.read_csv(run / "single_swaps.csv")
        self.assertTrue((swaps.qc_status == "PENDING").all())
        self.assertTrue((swaps.incoming_qc_flag != "EXPLICIT_ADEQUACY_REQUIRED").all())
        report = json.loads((run / "run.json").read_text())
        self.assertEqual(report["design"], "plan0009_v5_uniform_ge800_two_stage_hierarchical")
        self.assertEqual(report["single_swap_option_count"], len(swaps))
        self.assertEqual(pd.read_excel(workbook, sheet_name="pool").slide_id.tolist(),
                         pool.slide_id.tolist())
        self.assertEqual(pd.read_excel(workbook, sheet_name="eligibility").shape[0], 42)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            data = self.final_fixture("refuse_existing")
            v5.final_write(run, workbook, "stv", 42, data[0], data[1], data[7],
                           data[2], data[3], data[4], data[5], data[6], data[11],
                           {}, {}, data[10], data[8], data[9])

    def test_late_outputs_sources_and_anchor_mutation_prevent_final_complete(self):
        for mutation in ("workbook", "report", "swaps", "anchor", "source"):
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, "Source binding changed"):
                    self.write_final_fixture(f"mutated_{mutation}", mutation)
                self.assertFalse((self.scratch / f"mutated_{mutation}" / "COMPLETE").exists())

    def test_every_existing_final_artifact_is_preserved_without_completion(self):
        for name in ("pool.csv", "single_swaps.csv", "selected_ids.txt", "run.json",
                     "workbook", "COMPLETE"):
            with self.subTest(name=name):
                stem = f"collision_{name.replace('.', '_')}"
                (pop, ali, eligible, anchors, original, rectangular, selected, counts,
                 source, staged, eligibility, checks) = self.final_fixture(stem)
                run = self.scratch / stem
                workbook = self.scratch / f"{stem}.xlsx"
                collision = workbook if name == "workbook" else run / name
                collision.write_bytes(b"PREEXISTING_PARTIAL_ARTIFACT")
                anchor_sha = v5.v4.digest(run / "stage1_anchors.txt")
                with self.assertRaisesRegex(FileExistsError, "Final output already exists"):
                    v5.final_write(run, workbook, "stv", len(pop), pop, ali, counts, eligible,
                                   anchors, original, rectangular, selected, checks,
                                   {"objective": len(anchors)}, {"objective": 1},
                                   eligibility, source, staged)
                self.assertEqual(collision.read_bytes(), b"PREEXISTING_PARTIAL_ARTIFACT")
                self.assertEqual(v5.v4.digest(run / "stage1_anchors.txt"), anchor_sha)
                if name != "COMPLETE":
                    self.assertFalse((run / "COMPLETE").exists())
                for output_name in ("pool.csv", "single_swaps.csv", "selected_ids.txt", "run.json"):
                    if run / output_name != collision:
                        self.assertFalse((run / output_name).exists())

    def test_creation_race_at_exclusive_pool_open_preserves_competitor(self):
        (pop, ali, eligible, anchors, original, rectangular, selected, counts,
         source, staged, eligibility, checks) = self.final_fixture("exclusive_race")
        run = self.scratch / "exclusive_race"
        workbook = self.scratch / "exclusive_race.xlsx"
        pool_file = run / "pool.csv"
        original_open = Path.open
        raced = []

        def create_between_preflight_and_open(path, mode="r", *args, **kwargs):
            if path == pool_file and mode == "x" and not raced:
                raced.append(True)
                pool_file.write_bytes(b"RACER_OWNED_POOL")
            return original_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", create_between_preflight_and_open):
            with self.assertRaises(FileExistsError):
                v5.final_write(run, workbook, "stv", len(pop), pop, ali, counts, eligible,
                               anchors, original, rectangular, selected, checks,
                               {"objective": len(anchors)}, {"objective": 1},
                               eligibility, source, staged)
        self.assertEqual(raced, [True])
        self.assertEqual(pool_file.read_bytes(), b"RACER_OWNED_POOL")
        self.assertTrue((run / "STAGE1_COMPLETE").exists())
        self.assertFalse((run / "COMPLETE").exists())
        self.assertFalse(workbook.exists())

    def test_generated_matrix_anchor_and_source_postbinding_stage1(self):
        base, matrix, ali, candidates, counts = self.small()
        filtered = matrix.loc[list("ABCDEFG"), list("ABCDEFG")]
        nb = v5.neighborhoods(filtered)
        anchors, witness, info = v5.solve_stage1(nb, self.scratch)
        pop = base.loc[list("ABCDEFG")]
        eligible = pd.DataFrame({"slide_id": base.slide_id.tolist(), "quantity_eligible": [True] * 7 + [False],
                                 "final_candidate": [True] * 7 + [False]})
        source = self.source("matrix_source.txt")
        run = self.scratch / "matrix_source_stage1"
        bound = v5.stage1_write(run, pop, filtered, counts.loc[list(pop.index)], eligible,
                                anchors, witness, info, nb, source)
        matrix_path = run / "filtered_distance_matrix.pkl"
        saved = pd.read_pickle(matrix_path)
        saved.iat[0, 1] = 0.001
        saved.to_pickle(matrix_path)
        with self.assertRaisesRegex(ValueError, "Source binding changed"):
            v5.v4.verify_bindings(bound)
        anchors_file = run / "stage1_anchors.txt"
        anchors_file.write_text("different\n")
        with self.assertRaisesRegex(ValueError, "Source binding changed"):
            v5.v4.verify_bindings(bound)
        self.assertFalse((run / "COMPLETE").exists())

    def test_late_count_source_mutation_blocks_stage1_receipt(self):
        base, matrix, ali, candidates, counts = self.small()
        filtered = matrix.loc[list("ABCDEFG"), list("ABCDEFG")]
        nb = v5.neighborhoods(filtered)
        anchors, witness, info = v5.solve_stage1(nb, self.scratch)
        source = self.source("count_source.csv")
        path = Path(next(iter(source)))
        original_verify = v5.v4.verify_bindings
        run = self.scratch / "late_count_source_stage1"
        manifest = pd.DataFrame({"slide_id": base.slide_id.tolist(),
                                 "quantity_eligible": [True] * 7 + [False]})

        def mutate(current):
            original_verify(current)
            if current is source:
                path.write_text("late count-source mutation")

        with patch.object(v5.v4, "verify_bindings", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "Source binding changed"):
                v5.stage1_write(run, base.loc[list("ABCDEFG")], filtered,
                                counts.loc[list("ABCDEFG")], manifest,
                                anchors, witness, info, nb, source)
        self.assertFalse((run / "STAGE1_COMPLETE").exists())
        self.assertFalse((run / "COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
