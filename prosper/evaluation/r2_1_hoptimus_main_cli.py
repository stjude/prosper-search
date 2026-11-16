"""Bound CPU evaluation of saved H-Optimus main-grid ranks and distances."""

from __future__ import annotations

import argparse
import io
import itertools
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import prosper
import prosper.evaluation as evaluation
from . import feature_bank_audit as legacy
from . import r2_1_hoptimus_io as bound
from . import r2_1_hoptimus_main as kernel


METHODS = bound.METHODS
CUTOFFS = (1, 3, 5, 10)
ENDPOINTS = ("hit", "precision")
METADATA_HEADER = ("Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject")
INVENTORY_HEADER = ("bank", "slide_id", "canonical_tiles", "feature_rows",
                    "dtype", "feature_path", "resolved_path", "size", "sha256")
RANK_HEADER = ("query", "true_label", "rank", "neighbour", "neighbour_label", "distance")
POPULATION_HEADER = ("query_index", "reference_index", "metadata_index",
                     "slide_id", "patient_id", "diagnostic_group")
QUERY_HEADER = (
    "method", "query_index", "query", "patient_id", "diagnostic_group", "k", "status",
    "returned_depth", "short_list", "conditional_hit", "conditional_precision",
    "conditional_reason", "coverage", "hit_yield")
INDEX_HEADER = ("column", "row_kind", "method", "arm_a", "arm_b",
                "endpoint", "k", "level", "group")
ESTIMATE_HEADER = INDEX_HEADER + (
    "point", "point_status", "point_reason", "ci_low", "ci_high", "ci_status",
    "ci_reason", "n_queries", "n_patients", "zero_support_replicates")
TIE_HEADER = (
    "method", "query_index", "query", "patient_id", "diagnostic_group", "k",
    "boundary_distance", "strict_count", "strict_relevant", "block_count",
    "block_relevant", "slots", "hit_lower", "hit_expected", "hit_upper",
    "precision_lower", "precision_expected", "precision_upper")
POLICY_KEYS = ("method_order", "cutoff_order", "primary_cutoff", "endpoint_order",
               "group_order", "patient_order", "bootstrap", "contrast_pairs", "tie_policy")
DETAIL_KEYS = ("coverage", "returned_queries_per_method", "matrix_diagnostics", "tie_summaries")
MAIN_LIMITATIONS = tuple(bound.LIMITATIONS) + (
    "Coverage is one; intended and returned main populations are identical.",
    "Nominal intervals condition on supplied rankings and the fixed reference library.",
    "Tie bounds/expectations are sensitivity, not sampling intervals or winner evidence.",
    "Exclusions use patient identities across labels, never sentinel thresholds.",
    "Unused nonfinite distances are counted, not ranked; eligible 1e6 values are retained.",
    "Used 1e6 entries, if nonzero, require explicit actual-output disposition.",
)


def _int_cell(value, context, minimum=0):
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise bound.BoundError(f"{context}: invalid integer") from exc
    bound.require(str(result) == value and result >= minimum, f"{context}: noncanonical integer")
    return result


def _float_cell(value, context):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise bound.BoundError(f"{context}: invalid float") from exc
    bound.require(math.isfinite(result), f"{context}: nonfinite float")
    return result


def _metadata(spec, sources):
    rows, ids = [], set()
    for index, row in enumerate(bound.csv_rows(sources["metadata"], METADATA_HEADER)):
        slide = bound.text(row["Slide.Scan.File"], "metadata slide")
        patient = bound.text(row["UID.Subject"], "metadata patient")
        group = bound.text(row["Tumor.group"], "metadata group")
        bound.require(slide not in ids, "duplicate metadata slide")
        ids.add(slide)
        rows.append((index, slide, patient, group))
    target = [r for r in rows if r[3] != "MISC"]
    groups = tuple(sorted({r[3] for r in target}))
    patients = {}
    for _, _, p, g in target:
        patients.setdefault(p, set()).add(g)
    counts = dict(raw_slides=len(rows), target_slides=len(target), patients=len(patients),
                  groups=len(groups), cross_label_patients=sum(len(v) > 1 for v in patients.values()))
    bound.require(counts == spec["expected"], "metadata target/patient/group counts mismatch")
    summary = bound.strict_json(sources["stage0_summary"])
    banks = summary.get("declarations", {}).get("banks", [])
    bank_names = [b["name"] for b in banks]
    hopt = [b for b in banks if b["name"] == "h_optimus_1"]
    bound.require(len(bank_names) == len(set(bank_names)) and len(hopt) == 1
                  and type(hopt[0].get("feature_dim")) is int
                  and hopt[0]["feature_dim"] == 1536, "Stage0 H-Optimus width/declaration mismatch")
    inventory = set()
    totals = {b: 0 for b in bank_names}
    for row in bound.csv_rows(sources["stage0_inventory"], INVENTORY_HEADER):
        key = row["bank"], row["slide_id"]
        bound.require(row["bank"] in totals and row["slide_id"] in ids
                      and key not in inventory, "Stage0 inventory missing/extra/duplicate identity")
        inventory.add(key)
        tiles = _int_cell(row["canonical_tiles"], "canonical tiles", 1)
        bound.require(_int_cell(row["feature_rows"], "feature rows", 1) == tiles
                      and row["dtype"] in ("float16", "float32"), "inventory row count/dtype mismatch")
        _int_cell(row["size"], "feature size", 1)
        bound.digest(row["sha256"], "feature inventory")
        bound.text(row["feature_path"], "feature path")
        bound.text(row["resolved_path"], "resolved feature path")
        totals[row["bank"]] += tiles
    bound.require(inventory == set(itertools.product(bank_names, ids))
                  and summary.get("inventory_rows") == len(inventory)
                  and summary.get("bank_tile_totals") == totals, "Stage0 inventory totals mismatch")
    return rows, target, groups, counts


def _axis(axis, context):
    result = []
    for value in axis:
        if isinstance(value, str):
            result.append(bound.text(value, context))
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            result.append(str(int(value)))
        else:
            raise bound.BoundError(f"{context}: axis IDs must be exact strings or integral non-bools")
    bound.require(len(result) == len(set(result)), f"{context}: duplicate converted axis IDs")
    return result


def _ranked(data, target):
    qindex = {r[1]: i for i, r in enumerate(target)}
    n = len(target)
    neighbours = np.full((n, 10), -1, dtype=np.int64)
    distances = np.empty((n, 10), dtype=np.float64)
    seen = np.zeros((n, 10), dtype=bool)
    for row in bound.csv_rows(data, RANK_HEADER, compressed=True, max_expanded=bound.RANK_LIMIT):
        bound.require(row["query"] in qindex and row["neighbour"] in qindex,
                      "rank has missing/extra/ineligible query or neighbour")
        i, j = qindex[row["query"]], qindex[row["neighbour"]]
        rank = _int_cell(row["rank"], "rank", 1)
        bound.require(rank <= 10 and not seen[i, rank - 1], "duplicate/out-of-range rank")
        bound.require(row["true_label"] == target[i][3]
                      and row["neighbour_label"] == target[j][3], "rank labels disagree with metadata")
        bound.require(target[i][2] != target[j][2], "same-patient rank across labels")
        seen[i, rank - 1] = True
        neighbours[i, rank - 1] = j
        distances[i, rank - 1] = _float_cell(row["distance"], "rank distance")
    bound.require(seen.all(), "incomplete ranks: every target requires ranks 1-10")
    bound.require(all(len(set(row)) == 10 for row in neighbours), "duplicate ranked neighbour")
    return neighbours, distances


def _ties(method, i, target, scores, relevant):
    boundary = np.sort(scores, kind="mergesort")[4]
    before, equal = scores < boundary, scores == boundary
    c = int(np.count_nonzero(relevant & before))
    strict = int(before.sum())
    n, r, slots = int(equal.sum()), int(np.count_nonzero(relevant & equal)), 5 - strict
    low, high = c + max(0, slots - (n - r)), c + min(slots, r)
    hit_expected = (1.0 if c else
                    1 - (math.comb(n - r, slots) / math.comb(n, slots) if n - r >= slots else 0))
    return (
        method, i, target[i][1], target[i][2], target[i][3], 5, float(boundary),
        strict, c, n, r, slots, int(low > 0), hit_expected, int(high > 0),
        low / 5, (c + slots * r / n) / 5, high / 5)


def _matrix(method, sources, raw, target):
    matrix = legacy._pickle(sources[f"matrix_{method}"], f"trusted local matrix {method}")
    bound.require(isinstance(matrix, pd.DataFrame), "matrix must be a pandas DataFrame")
    row_ids, col_ids = _axis(matrix.index, "matrix rows"), _axis(matrix.columns, "matrix columns")
    raw_ids = {r[1] for r in raw}
    bound.require(set(row_ids) == set(col_ids) == raw_ids
                  and matrix.shape == (len(raw), len(raw)), "matrix raw axes mismatch")
    values = matrix.to_numpy(copy=False)
    bound.require(values.dtype.kind == "f" and values.ndim == 2, "matrix must be floating numeric")
    rows, cols = {q: i for i, q in enumerate(row_ids)}, {q: i for i, q in enumerate(col_ids)}
    positions = np.array([cols[r[1]] for r in target], dtype=np.int64)
    labels = np.array([r[3] for r in target], dtype=object)
    patients = np.array([r[2] for r in target], dtype=object)
    neighbours, saved_distances = _ranked(sources[f"ranks_{method}"], target)
    relevance = labels[neighbours] == labels[:, None]
    diagnostic = {
        "used_pairs": 0, "excluded_same_patient_pairs": 0,
        "unused_nonfinite_entries": sum(int(np.count_nonzero(~np.isfinite(row))) for row in values),
        "used_1e6_entries": 0,
    }
    tie_rows = []
    for i, query in enumerate(target):
        eligible = patients != query[2]
        candidates = np.flatnonzero(eligible)
        scores = values[rows[query[1]], positions[eligible]]
        bound.require(len(scores) >= 10, "fewer than ten eligible main references")
        bound.require(np.isfinite(scores).all(), "nonfinite distance on used eligible pair")
        ordered = np.argsort(scores, kind="mergesort")
        expected = candidates[ordered[:10]]
        bound.require(np.array_equal(neighbours[i], expected), "saved ranking/matrix stable-order mismatch")
        bound.require(all(abs(float(a) - float(b)) <= 1e-14
                          for a, b in zip(scores[ordered[:10]], saved_distances[i])),
                      "CSV/matrix distance disagreement")
        diagnostic["used_pairs"] += len(scores)
        diagnostic["excluded_same_patient_pairs"] += int(np.count_nonzero(~eligible))
        diagnostic["used_1e6_entries"] += int(np.count_nonzero(scores == 1e6))
        tie_rows.append(_ties(method, i, target, scores, labels[eligible] == query[3]))
    return relevance, tie_rows, diagnostic


def _query_rows(target, relevances):
    for method in METHODS:
        for i, query in enumerate(target):
            for k in CUTOFFS:
                outcome = kernel.query_metrics(relevances[method][i], k, "returned")
                yield (
                    method, i, query[1], query[2], query[3], k, outcome.status,
                    outcome.returned_depth, outcome.short_list, outcome.conditional_hit,
                    outcome.conditional_precision, outcome.conditional_reason,
                    outcome.coverage, outcome.hit_yield)


def _metric_vectors(target, relevances):
    return {
        (method, k, endpoint): np.array([
            (float(np.any(row[:k])) if endpoint == "hit" else np.count_nonzero(row[:k]) / k)
            for row in relevances[method]], dtype=np.float64)
        for method in METHODS for k in CUTOFFS for endpoint in ENDPOINTS
    }


def _iter_results(target, groups, vectors, draws):
    qids, pids, labels = [list(x) for x in zip(*[(r[1], r[2], r[3]) for r in target])]
    included = np.ones(len(target), dtype=bool)
    for method in METHODS:
        for k in CUTOFFS:
            for endpoint in ENDPOINTS:
                result = kernel.summarize_metric(
                    qids, pids, labels, groups, vectors[method, k, endpoint], included, draws)
                for group, estimate in zip((*groups, ""), (*result.group_estimates, result.macro)):
                    yield ("estimate", method, "", "", endpoint, k,
                           "group" if group else "macro", group), estimate
    for i, j in itertools.combinations(range(4), 2):
        a, b = METHODS[j], METHODS[i]
        for endpoint in ENDPOINTS:
            result = kernel.summarize_paired(
                qids, pids, labels, groups, vectors[a, 5, endpoint],
                vectors[b, 5, endpoint], included, draws).difference
            for group, estimate in zip((*groups, ""), (*result.group_estimates, result.macro)):
                yield ("contrast", "", a, b, endpoint, 5,
                       "group" if group else "macro", group), estimate


def _estimate_row(column, key, result):
    return (column, *key, result.point, result.point_status, result.point_reason,
            result.ci_low, result.ci_high, result.ci_status, result.ci_reason,
            result.n_queries, result.n_patients, result.zero_support_replicates)


def _npy(values, dtype):
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(values, dtype=dtype),
                              version=(1, 0), allow_pickle=False)
    return buffer.getvalue()


def _load_npy(data, shape, dtype, name):
    bound.require(data[:8] == b"\x93NUMPY\x01\x00", f"{name}: requires NPY version 1.0")
    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_1_0(stream)
        bound.require(version == (1, 0) and actual_shape == shape and not fortran
                      and actual_dtype.str == np.dtype(dtype).str and not actual_dtype.hasobject
                      and stream.tell() + math.prod(shape) * actual_dtype.itemsize == len(data),
                      f"{name}: invalid header shape/dtype/order/length")
        stream.seek(0)
        values = np.load(stream, allow_pickle=False)
    except (ValueError, TypeError, EOFError) as exc:
        raise bound.BoundError(f"{name}: invalid numeric NPY") from exc
    bound.require(stream.tell() == len(data) and values.shape == shape
                  and values.dtype.str == np.dtype(dtype).str and values.flags.c_contiguous,
                  f"{name}: trailing bytes, wrong shape/dtype/order")
    return values


def _tie_summaries(rows, groups):
    result = []
    for method in METHODS:
        selected = [row for row in rows if row[0] == method]
        for endpoint, start in (("hit", 12), ("precision", 15)):
            group_values = []
            for group in groups:
                members = [row for row in selected if row[4] == group]
                values = [float(np.mean([row[start + offset] for row in members])) for offset in range(3)]
                group_values.append(values)
                result.append(dict(method=method, endpoint=endpoint, level="group", group=group,
                                   lower=values[0], expected=values[1], upper=values[2]))
            values = np.mean(group_values, axis=0)
            result.append(dict(method=method, endpoint=endpoint, level="macro", group="",
                               lower=float(values[0]), expected=float(values[1]), upper=float(values[2])))
    return result


def _policy(groups, patients):
    return {
        "method_order": list(METHODS), "cutoff_order": list(CUTOFFS), "primary_cutoff": 5,
        "endpoint_order": list(ENDPOINTS), "group_order": list(groups), "patient_order": list(patients),
        "bootstrap": {"count": 5000, "seed": 123, "generator": "PCG64",
                      "quantiles": [0.025, 0.975], "quantile_method": "linear"},
        "contrast_pairs": [dict(arm_a=METHODS[j], arm_b=METHODS[i])
                           for i, j in itertools.combinations(range(4), 2)],
        "tie_policy": "metadata_order_primary_exact_distance_sensitivity",
    }


def _as_strings(row):
    return tuple(bound._cell(value) for value in row)


def _equal_csv(data, header, expected, *, compressed=False):
    parsed = bound.csv_rows(data, header, compressed=compressed)
    sentinel = object()
    for actual, wanted in itertools.zip_longest(parsed, expected, fillvalue=sentinel):
        bound.require(actual is not sentinel and wanted is not sentinel
                      and tuple(actual[k] for k in header) == _as_strings(wanted),
                      "serialized CSV content/order/length reconciliation failed")


def validate_artifacts(artifacts, spec, target, groups, relevances, tie_rows, diagnostics):
    """Reconstruct numerical summaries from persisted draws and query outcomes."""
    bound.require(set(artifacts) == bound.ARTIFACTS["hoptimus_main"], "main artifact set mismatch")
    _equal_csv(artifacts["populations.csv"], POPULATION_HEADER,
               ((i, i, *r) for i, r in enumerate(target)))
    _equal_csv(artifacts["query_metrics.csv.gz"], QUERY_HEADER,
               _query_rows(target, relevances), compressed=True)
    vectors = {key: np.empty(len(target), dtype=np.float64)
               for key in itertools.product(METHODS, CUTOFFS, ENDPOINTS)}
    for row in bound.csv_rows(artifacts["query_metrics.csv.gz"], QUERY_HEADER, compressed=True):
        i, k = _int_cell(row["query_index"], "query index"), _int_cell(row["k"], "cutoff")
        for endpoint, column in (("hit", "conditional_hit"), ("precision", "conditional_precision")):
            vectors[row["method"], k, endpoint][i] = _float_cell(row[column], column)
    patients = tuple(sorted({r[2] for r in target}))
    weights = _load_npy(artifacts["patient_draws.npy"], (5000, len(patients)), "<i8", "draws")
    bound.require(np.array_equal(weights, kernel.make_patient_draws(patients).multiplicities),
                  "persisted patient draws differ from fixed policy")
    draws = kernel.PatientDraws(patients, weights)
    c = 44 * (len(groups) + 1)
    replicates = _load_npy(artifacts["replicates.npy"], (5000, c), "<f8", "replicates")
    valid = _load_npy(artifacts["replicate_valid.npy"], (5000, c), "|b1", "validity")
    bound.require(np.isfinite(replicates[valid]).all() and np.isnan(replicates[~valid]).all(),
                  "replicate finite/NaN validity mismatch")
    indices, estimates, contrasts = [], [], []
    for column, (key, result) in enumerate(_iter_results(target, groups, vectors, draws)):
        indices.append((column, *key))
        (estimates if key[0] == "estimate" else contrasts).append(_estimate_row(column, key, result))
        bound.require(np.array_equal(valid[:, column], result.valid)
                      and np.array_equal(replicates[:, column], result.replicates, equal_nan=True),
                      "persisted replicate values/support masks disagree")
    _equal_csv(artifacts["replicate_index.csv"], INDEX_HEADER, indices)
    _equal_csv(artifacts["estimates.csv"], ESTIMATE_HEADER, estimates)
    _equal_csv(artifacts["contrasts.csv"], ESTIMATE_HEADER, contrasts)
    _equal_csv(artifacts["tie_sensitivity.csv.gz"], TIE_HEADER, tie_rows, compressed=True)
    decoded_ties = []
    for row in bound.csv_rows(artifacts["tie_sensitivity.csv.gz"], TIE_HEADER, compressed=True):
        decoded_ties.append(tuple(
            row[k] if k in ("method", "query", "patient_id", "diagnostic_group")
            else _float_cell(row[k], k) for k in TIE_HEADER))
    summary = bound.strict_json(artifacts["summary.json"])
    bound.keys(summary, ("schema_version", "run_id", "task", "counts",
                         "policy", "details", "limitations"), "main summary")
    bound.keys(summary["policy"], POLICY_KEYS, "main policy")
    bound.keys(summary["details"], DETAIL_KEYS, "main details")
    expected_summary = {
        "schema_version": 1, "run_id": spec["run_id"], "task": "hoptimus_main",
        "counts": spec["expected"], "policy": _policy(groups, patients),
        "details": {
            "coverage": 1.0, "returned_queries_per_method": {m: len(target) for m in METHODS},
            "matrix_diagnostics": diagnostics, "tie_summaries": _tie_summaries(decoded_ties, groups),
        },
        "limitations": list(MAIN_LIMITATIONS),
    }
    bound.require(bound.json_bytes(summary) == bound.json_bytes(expected_summary),
                  "main summary type-sensitive semantic readback mismatch")


def execute(spec, sources):
    """Consume declared bytes only and return semantically read-back artifacts."""
    bound._schema(spec, "hoptimus_main")
    bound.require(set(sources) == set(spec["sources"]), "main source roles mismatch")
    raw, target, groups, counts = _metadata(spec, sources)
    relevances, ties, diagnostics = {}, [], {}
    for method in METHODS:
        relevance, rows, diagnostic = _matrix(method, sources, raw, target)
        relevances[method] = relevance
        ties.extend(rows)
        diagnostics[method] = diagnostic
    draws = kernel.make_patient_draws([r[2] for r in target])
    vectors = _metric_vectors(target, relevances)
    c = 44 * (len(groups) + 1)
    replicates = np.empty((5000, c), dtype="<f8")
    valid = np.empty((5000, c), dtype=np.bool_)
    indices, estimates, contrasts = [], [], []
    for column, (key, result) in enumerate(_iter_results(target, groups, vectors, draws)):
        indices.append((column, *key))
        (estimates if key[0] == "estimate" else contrasts).append(_estimate_row(column, key, result))
        replicates[:, column], valid[:, column] = result.replicates, result.valid
    summary = {
        "schema_version": 1, "run_id": spec["run_id"], "task": "hoptimus_main", "counts": counts,
        "policy": _policy(groups, draws.patient_ids),
        "details": {
            "coverage": 1.0, "returned_queries_per_method": {m: len(target) for m in METHODS},
            "matrix_diagnostics": diagnostics, "tie_summaries": _tie_summaries(ties, groups),
        },
        "limitations": list(MAIN_LIMITATIONS),
    }
    artifacts = {
        "populations.csv": bound.csv_bytes(POPULATION_HEADER,
                                          ((i, i, *r) for i, r in enumerate(target))),
        "query_metrics.csv.gz": bound.csv_bytes(QUERY_HEADER, _query_rows(target, relevances), compressed=True),
        "replicate_index.csv": bound.csv_bytes(INDEX_HEADER, indices),
        "estimates.csv": bound.csv_bytes(ESTIMATE_HEADER, estimates),
        "contrasts.csv": bound.csv_bytes(ESTIMATE_HEADER, contrasts),
        "tie_sensitivity.csv.gz": bound.csv_bytes(TIE_HEADER, ties, compressed=True),
        "summary.json": bound.json_bytes(summary),
        "patient_draws.npy": _npy(draws.multiplicities, "<i8"),
        "replicates.npy": _npy(replicates, "<f8"),
        "replicate_valid.npy": _npy(valid, "|b1"),
    }
    del replicates, valid, vectors, draws
    validate_artifacts(artifacts, spec, target, groups, relevances, ties, diagnostics)
    return artifacts


def code_paths():
    return {
        "runner": __file__, "kernel": kernel.__file__, "io": bound.__file__,
        "legacy_io": legacy.__file__, "package_init": prosper.__file__,
        "evaluation_init": evaluation.__file__,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("spec", "spec-sha256", "out", "out-root"):
        parser.add_argument(f"--{option}", required=True)
    args = parser.parse_args(argv)
    try:
        bound.run_bound(args.spec, args.spec_sha256, args.out, args.out_root,
                        task="hoptimus_main", code_paths=code_paths(), execute=execute)
    except Exception as exc:
        print(f"H-Optimus main FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"H-Optimus main COMPLETE: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
