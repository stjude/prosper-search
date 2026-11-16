#!/usr/bin/env python3
"""E1-BAYHO-P: verify frozen held-out inputs and recount the authorized 2,604-query primary analysis.

See [analysis plan], E1-BAYHO-P. No fitting, OT, ROI search or upstream write.
The retained Bayesian implementation produces the results; independently
sorted raw rows and scalar power weights verify them before release.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plurality_prevalence(labels, prevalence):
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], -prevalence[label], label))


def scalar_probabilities(distances, labels, classes, k):
    span = float(distances[k] - distances[0])
    weights = [1.0 if span < 1e-10 else 20.0 ** ((float(distances[0]) - float(d)) / span)
               for d in distances[:k]]
    denominator = math.fsum(weights)
    return np.array([math.fsum(w for w, label in zip(weights, labels[:k]) if label == group)
                     / denominator for group in classes])


def standard_metrics(y, prediction):
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction))}


def error_comparison(y, bayes_prediction, vote, probabilities):
    corrected = (bayes_prediction == y) & (vote != y)
    introduced = (bayes_prediction != y) & (vote == y)
    top = np.sort(probabilities, axis=1)[:, -2:]
    ratio = (top[:, 1] + 0.001) / (top[:, 0] + 0.001)
    return {"corrected": int(corrected.sum()), "introduced": int(introduced.sum()),
            "ratio_median_corrected": float(np.median(ratio[corrected])),
            "ratio_median_introduced": float(np.median(ratio[introduced])),
            "ratio_mann_whitney_p": float(mannwhitneyu(ratio[corrected], ratio[introduced],
                                                     alternative="two-sided").pvalue)}, corrected, introduced, ratio


def run(destination, out):
    prior = out / "e1_bayes_heldout_filter_sensitivity_20261114/run_002"
    tf = out / "r2_1_filter_tf2_20261016"
    checks, bindings = {}, {}

    def bind(path, expected=None):
        path = Path(path)
        old_mount = Path("/PROSPER_WORKSPACE/prosper_revision")
        if path.is_relative_to(old_mount):
            path = ROOT / path.relative_to(old_mount)
        actual = digest(path)
        assert expected is None or actual == expected, f"Hash mismatch: {path}"
        bindings[str(path)] = actual
        return path

    prior_binding = json.loads(bind(prior / "input_bindings.json").read_text())
    prior_results = json.loads(bind(prior / "results.json").read_text())
    used = ("matrix_heldout_both_sided_b100", "population", "query_assignment",
            "ranked_lists", "library_map", "case_metadata")
    inputs = {key: bind(prior_binding[key]["path"], prior_binding[key]["sha256"]) for key in used}
    source = bind(prior / "e1_bayes_heldout_filter_sensitivity.py",
                  prior_results["env"]["script_sha256"])
    bind(ROOT / "scripts/e1_bayes_heldout_filter_sensitivity.py", digest(source))
    spec = importlib.util.spec_from_file_location("retained_bayesian", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    checks["retained_source_and_used_inputs_hash_match"] = True

    population = pd.read_csv(inputs["population"], dtype=str).sort_values("query_index", key=lambda s: s.astype(int))
    ids = population.slide_id.tolist()
    groups = population.diagnostic_group.to_numpy()
    patients = population.patient_id.to_numpy()
    classes = sorted(set(groups))
    prevalence = Counter(groups)
    assert len(ids) == len(set(ids)) == 2604 and len(classes) == 14 and len(set(patients)) == 1800
    library = pd.read_csv(inputs["library_map"], dtype=str).set_index("Slide.Scan.File")
    assert library.index.is_unique
    assert library.loc[ids, "Tumor.group"].tolist() == groups.tolist()
    assert library.loc[ids, "UID.Subject"].tolist() == patients.tolist()
    assert {c: int((library["Tumor.group"] == c).sum()) for c in classes} == dict(prevalence)
    # Freeze the frequency order from the reference population, before comparison outcomes.
    order = sorted(classes, key=lambda c: (-prevalence[c], c))
    checks["population_and_reference_prevalence"] = {"queries": len(ids), "patients": len(set(patients)),
                                                   "groups": len(classes), "order": order,
                                                   "group_counts": dict(prevalence)}

    assignment = pd.read_csv(inputs["query_assignment"], dtype=str).set_index("slide_id").loc[ids]
    assert assignment.patient_id.tolist() == patients.tolist()
    training = {}
    for slot in sorted(assignment.head_slot.unique()):
        head = tf / "s2/run_002/heads/uni2/b100" / slot
        complete = json.loads(bind(head / "COMPLETE.json").read_text())
        counts = json.loads(bind(head / "counts.json", complete["outputs"]["counts.json"]["sha256"]).read_text())
        bind(head / "head.pkl", complete["outputs"]["head.pkl"]["sha256"])
        assert counts["head_id"] == complete["head_id"] == f"uni2/b100/{slot}"
        training[slot] = set(counts["training_patients"])
    assert all(patient not in training[slot] for patient, slot in zip(patients, assignment.head_slot))
    full_patients = training["full"]
    expected_slots = [f"fold{fold}" if patient in full_patients else "full"
                      for patient, fold in zip(patients, assignment.patient_fold)]
    assert assignment.head_slot.tolist() == expected_slots
    assert all(assignment.library_version_both_sided == "v_" + assignment.head_slot)
    checks["all_query_patients_absent_from_assigned_training_head"] = {
        "checked": len(ids), "violations": 0, "head_slots": assignment.head_slot.value_counts().to_dict(),
        "limit": "Uses hash-bound executed training-patient records; earlier TF2 audit checked actual training tiles."
    }

    stem = "combined_similarity_matrix_tf2_uni2_b100_both_sided_ot_cosine"
    combined = inputs["matrix_heldout_both_sided_b100"].parent
    combine = json.loads(bind(combined / f"{stem}_combine_record.json").read_text())
    for suffix in ("_no_same_patient.pkl", "_rows.csv", "_rows.json"):
        bind(combined / f"{stem}{suffix}", combine["outputs"][f"{stem}{suffix}"]["sha256"])
    rows = pd.read_csv(combined / f"{stem}_rows.csv", dtype=str).set_index("query").loc[ids]
    assert rows.patient_id.tolist() == patients.tolist()
    assert rows.head_slot.tolist() == assignment.head_slot.tolist()
    assert rows.library_version.tolist() == assignment.library_version_both_sided.tolist()
    assert all(rows.query_head_id == "uni2/b100/" + rows.head_slot)
    assert all(rows.query_head_id == rows.library_head_id)
    assert all(rows.n_failed_pairs == "0") and combine["failed_pairs_total"] == 0
    matrix = pd.read_pickle(inputs["matrix_heldout_both_sided_b100"])
    matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)
    assert set(matrix.index) == set(matrix.columns) == set(ids) and matrix.shape == (2604, 2604)
    matrix = matrix.loc[ids, ids]
    same_patient = patients[:, None] == patients[None, :]
    x = matrix.to_numpy(dtype=float)
    assert not np.any(np.isfinite(x[same_patient]) & (x[same_patient] < 1e5))
    assert np.isfinite(x[~same_patient]).all() and (x[~same_patient] < 1e5).all()
    sample = {sid for _, sid, _ in original.CASES} | {"16738"}
    sample.update(rows.reset_index().groupby("head_slot", sort=True).first()["query"])
    for sid in sorted(sample):
        row = rows.loc[sid]
        pairs = pd.read_pickle(bind(row.row_file, row.row_sha256))
        raw = pd.Series(dict(pairs), dtype=float)
        assert set(raw.index) == set(ids)
        eligible = [ref for ref, patient in zip(ids, patients) if patient != row.patient_id]
        np.testing.assert_allclose(raw.loc[eligible].to_numpy(), matrix.loc[sid, eligible].to_numpy(),
                                   rtol=0, atol=1e-12)
    checks["matrix_row_heads_exclusions_and_raw_row_sample"] = {
        "all_row_heads_match": True, "same_patient_eligible_cells": 0,
        "raw_row_sample": sorted(sample), "failed_pairs": 0,
        "directionality": "Rows retain their query-specific head; no symmetrization."
    }

    conf = original.prepare(matrix, ids, groups.tolist(), patients.tolist(), classes)
    assert np.all(conf["n_finite"] == 26)
    raw_distances, raw_labels, raw_neighbors = [], [], []
    for sid, patient in zip(ids, patients):
        eligible = matrix.loc[sid].loc[patients != patient].sort_values(kind="mergesort").iloc[:26]
        raw_distances.append(eligible.to_numpy())
        raw_neighbors.append(eligible.index.tolist())
        raw_labels.append(library.loc[eligible.index, "Tumor.group"].tolist())
    raw_distances = np.array(raw_distances)
    raw_labels = np.array(raw_labels)
    raw_neighbors = np.array(raw_neighbors)
    np.testing.assert_allclose(raw_distances, conf["dist"], rtol=0, atol=0)
    assert np.array_equal(raw_neighbors, np.array(ids)[conf["nbr"]])
    ranked = pd.read_csv(inputs["ranked_lists"], dtype={"query": str, "neighbour": str})
    ranked = ranked[(ranked.encoder == "uni2") & (ranked.arm == "both_filtered")
                    & (ranked.method == "ot_cosine") & (ranked.budget == 100)]
    assert len(ranked) == 26040
    for sid, table in ranked.groupby("query"):
        i = ids.index(sid)
        table = table.sort_values("rank")
        assert table.neighbour.tolist() == raw_neighbors[i, :10].tolist()
        np.testing.assert_allclose(table.distance, raw_distances[i, :10], rtol=0, atol=1e-12)
    checks["independent_rankings_and_retained_top_ten"] = {"queries": len(ids), "ranks_checked": 26,
                                                         "retained_top_ten_rows": len(ranked), "mismatches": 0}

    submitted = ROOT / "docs/prosper_submission/supplementary/supplementary_tables"
    old_metrics = pd.read_excel(bind(next(submitted.glob("Table_S9_*.xlsx")))).set_index("N")
    old_checks = {}
    for k, nl, np_ in ((5, 10, 11), (18, 12, 13)):
        labels = pd.read_excel(bind(next(submitted.glob(f"Table_S{nl}_*.xlsx"))))
        probs = pd.read_excel(bind(next(submitted.glob(f"Table_S{np_}_*.xlsx")))).set_index("slides")
        old_classes = sorted(set(library["Tumor.group"]) - {"MISC"})
        old_p = probs.loc[labels.slides, old_classes].to_numpy()
        truth = labels.disease.to_numpy()
        pred = np.array(old_classes)[old_p.argmax(axis=1)]
        freq = library["Tumor.group"].value_counts().to_dict()
        vote = np.array([plurality_prevalence(row, freq)
                         for row in labels[[f"N{j}" for j in range(k)]].to_numpy()])
        for rule, prediction in (("Bayesian", pred), ("no tie break", vote)):
            metrics = standard_metrics(truth, prediction)
            for name, prefix in (("accuracy", "accuracy"), ("macro_f1", "F1"),
                                 ("balanced_accuracy", "balanced accuracy")):
                assert abs(metrics[name] - old_metrics.loc[f"{prefix}-{rule}", k]) < 1e-12
        comparison, good, bad, ratio = error_comparison(truth, pred, vote, old_p)
        assert np.array_equal(good, labels.candidate.eq("yes"))
        assert np.array_equal(bad, labels["reverse-candidate"].eq("yes"))
        np.testing.assert_allclose(ratio, labels.LR1t2, rtol=1e-12, atol=1e-12)
        old_checks[str(k)] = comparison | {"all_submitted_flags_equal": True}
    checks["historical_prevalence_comparator_reproduction"] = old_checks

    prior_queries = pd.read_csv(bind(prior / "per_query.csv.gz"), dtype={"query": str}).set_index("query").loc[ids]
    matched = prior_queries.in_common_set.eq(1).to_numpy()
    assert matched.sum() == 2603 and np.array(ids)[~matched].tolist() == ["16738"]
    prior_sweep = pd.read_csv(bind(prior / "bayesian_metrics_by_k.csv")).query("config == 'B'").set_index("k")
    prior_cases = pd.read_csv(bind(prior / "cases.csv"), dtype={"scan_id": str})
    metric_rows, comparisons, probability_tables, prediction_tables = [], {}, {}, {}
    worst_error = 0.0
    y = conf["y"]
    for k in original.K_SWEEP:
        p, ok = original.bayes(conf, k)
        assert ok.all()
        independent = np.stack([scalar_probabilities(d, label, classes, k)
                                for d, label in zip(raw_distances, raw_labels)])
        worst_error = max(worst_error, float(np.abs(p - independent).max()))
        np.testing.assert_allclose(p, independent, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
        assert np.isfinite(p).all() and (p >= 0).all()
        bayes_prediction = np.array(classes)[original.argmax_pred(p, ok)]
        vote = np.array([plurality_prevalence(label[:k], prevalence) for label in raw_labels])
        closest = np.array(classes)[original.plurality_pred(conf, k)[0]]
        for rule, prediction in (("bayesian_argmax", bayes_prediction),
                                 ("plurality_prevalence", vote), ("plurality_closest_rank", closest)):
            m = standard_metrics(groups, prediction)
            numeric_pred = np.array([classes.index(label) for label in prediction])
            check_m = original.metrics(y, numeric_pred, len(classes))
            assert all(abs(m[name] - check_m[name]) < 1e-12 for name in m)
            metric_rows.append({"k": k, "rule": rule, "n_queries": len(ids), **m})
        matched_metrics = standard_metrics(groups[matched], bayes_prediction[matched])
        assert all(abs(value - prior_sweep.loc[k, name]) < 1e-12 for name, value in matched_metrics.items())
        if k in (5, 18):
            assert bayes_prediction[matched].tolist() == prior_queries.loc[matched, f"pred_B_k{k}"].tolist()
            np.testing.assert_allclose(p.max(axis=1), prior_queries[f"top_prob_B_k{k}"], rtol=0, atol=1e-12)
            assert closest.tolist() == prior_queries[f"vote_B_k{k}"].tolist()
            table = pd.DataFrame({"query": ids, "patient": patients, "group": groups,
                                  "bayesian_prediction": bayes_prediction, "prevalence_prediction": vote,
                                  "closest_prediction": closest, "in_old_matched_subset": matched.astype(int)})
            comparisons[str(k)] = {}
            for rule, prediction in (("prevalence", vote), ("closest_rank", closest)):
                c, good, bad, ratio = error_comparison(groups, bayes_prediction, prediction, p)
                comparisons[str(k)][rule] = c
                table[f"{rule}_corrected"] = good.astype(int)
                table[f"{rule}_introduced"] = bad.astype(int)
                table["top_two_ratio"] = ratio
                independent_c = original.g_counts(y, np.array([classes.index(g) for g in bayes_prediction]),
                                                  np.array([classes.index(g) for g in prediction]), p)
                assert (c["corrected"], c["introduced"]) == (independent_c["corrected"], independent_c["introduced"])
                assert abs(c["ratio_mann_whitney_p"] - independent_c["mann_whitney_p"]) < 1e-12
            for _, sid, _ in original.CASES:
                expected = prior_cases[(prior_cases.config == "B") & (prior_cases.scan_id == sid) & (prior_cases.k == k)]
                expected_p = expected.set_index("group").probability.reindex(classes, fill_value=0).to_numpy()
                np.testing.assert_allclose(p[ids.index(sid)], expected_p, rtol=0, atol=1e-12)
            probability_tables[k] = pd.DataFrame(p, index=pd.Index(ids, name="query"), columns=classes)
            prediction_tables[k] = table
    checks["all_primary_probabilities_independent_scalar_check"] = {
        "queries": len(ids), "k_values": list(original.K_SWEEP), "max_absolute_difference": worst_error}
    checks["matched_2603_sweep_predictions_case_values"] = "All reproduced within 1e-12; one extra valid query retained."

    metrics = pd.DataFrame(metric_rows)
    summaries = {}
    for rule, table in metrics.groupby("rule", sort=False):
        best = table.loc[table.macro_f1.idxmax()]
        summaries[rule] = {"k_max_f1": int(best.k), "max_f1": float(best.macro_f1),
                           "f1_range": [float(table.macro_f1.min()), float(table.macro_f1.max())],
                           "all_k": table.k.tolist()}
    case_rows = prior_cases.loc[prior_cases.config.eq("B")].copy()
    case_rows["config"] = "heldout_primary_2604"
    neighbor_table = pd.DataFrame({
        "query": np.repeat(ids, 26), "rank": np.tile(np.arange(1, 27), len(ids)),
        "neighbor": raw_neighbors.ravel(), "neighbor_group": raw_labels.ravel(),
        "distance": raw_distances.ravel(),
    })
    metrics.to_csv(destination / "metrics_by_k.csv", index=False)
    case_rows.to_csv(destination / "cases.csv", index=False)
    neighbor_table.to_csv(destination / "neighbors_top26.csv.gz", index=False)
    for k, p in probability_tables.items():
        p.to_csv(destination / f"probabilities_k{k}.csv.gz")
        prediction_tables[k].to_csv(destination / f"predictions_k{k}.csv.gz", index=False)
    results = {
        "status": "VERIFIED", "population": checks["population_and_reference_prevalence"],
        "parameters": {"T": 20, "k_examples": [5, 18], "k_sweep": list(original.K_SWEEP),
                       "configuration": "UNI2, TF2 b100, query-head-matched both-sided library, OT cosine",
                       "same_patient_excluded": True, "bayesian_ties": "alphabetical",
                       "primary_vote_ties": "fixed reference-population frequency, alphabetical final fallback",
                       "test_scope": "unadjusted two-sided slide-level Mann-Whitney; not patient-independent"},
        "checks": checks, "sweep": summaries, "error_comparisons": comparisons,
        "extra_query": {str(k): prediction_tables[k].query("query == '16738'").to_dict("records")[0]
                        for k in (5, 18)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "host": platform.node()},
        "limits": [
            "Upstream fitting/masks/OT use accepted TF2 review; not rerun.",
            "Other annotated reference patients can have in-sample masks under the query-held-out head.",
            "Post-result adoption; descriptive k sweep on the same evaluation queries, not validated tuning.",
            "Probabilities uncalibrated; reference composition affects estimates.",
            "Clinical labels/ROI identities/pathology not re-adjudicated; no new ROI results.",
            "Historical k=2 discrepancy remains; no historical result overwritten.",
        ],
    }
    (destination / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (destination / "input_bindings.json").write_text(json.dumps(bindings, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="run_001")
    args = parser.parse_args()
    if not args.run.startswith("run_") or not args.run[4:].isdigit():
        parser.error("--run must be run_<digits>")
    out = Path(os.environ.get("PROSPER_OUT", ROOT / "prosper_revision_output"))
    destination = out / "e1_bayes_heldout_primary_20261114" / args.run
    destination.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    for path in (Path(__file__), ROOT / "tests/test_e1_bayes_heldout_filter_sensitivity.py",
                 ROOT / "scripts/e1_bayes_heldout_filter_sensitivity.py"):
        shutil.copy2(path, destination / path.name)
    try:
        results = run(destination, out)
    except Exception as error:
        (destination / "FAILED.json").write_text(json.dumps(
            {"type": type(error).__name__, "error": str(error), "elapsed_seconds": time.monotonic() - start}, indent=2) + "\n")
        raise
    files = sorted(p for p in destination.iterdir() if p.is_file())
    (destination / "SHA256SUMS.txt").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files))
    print(json.dumps({"status": results["status"], "output": str(destination),
                      "seconds": time.monotonic() - start, "sweep": results["sweep"],
                      "error_comparisons": results["error_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
