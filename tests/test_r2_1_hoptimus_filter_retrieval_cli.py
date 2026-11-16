"""Owned synthetic pilot seams; reader-bound CLI integration is separately gated."""
from dataclasses import FrozenInstanceError, asdict, dataclass, replace
import copy
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

from prosper.evaluation import r2_1_hoptimus_filter_retrieval_cli as cli


def request(name="pair", method="ot_cosine", selection="filtered", budget=5, q=0, r=1):
    return {"request_id": name, "query_index": q, "reference_index": r, "query_fold": 0,
            "method": method, "selection": selection,
            "budget_percent": budget if selection == "filtered" else None}


def row_request(name, method="yottixel", selection="filtered", budget=5, q=0):
    value = request(name, method, selection, budget, q)
    del value["reference_index"]
    return value


def spec(pairs=None, rows=None):
    pairs = [request()] if pairs is None else pairs
    rows = [] if rows is None else rows
    queries = list(dict.fromkeys(r["query_index"] for r in pairs+rows))
    baseline = any(r["selection"] != "filtered" for r in pairs) or bool(rows)
    return {"schema_version": 1, "task": cli.TASK, "run_id": "synthetic-only",
        "code_files": {}, "mosaic_bundle_assets": {}, "mosaic_expected": {
            "counts": {"slides": 3}}, "baseline_assets": {} if baseline else None,
        "baseline_expected": {} if baseline else None, "runtime_assets": {},
        "query_indices": queries, "pair_requests": pairs, "row_requests": rows,
        "limits": dict(cli.LIMITS)}


@dataclass(frozen=True)
class Slide:
    slide_index: int
    metadata_index: int
    slide_id: str
    patient_id: str
    diagnostic_group: str
    slide_fold: int
    tile_start: int
    tile_stop: int
    feature_rows: int


class Backing:
    def __init__(self, array):
        self.encoded = array.tobytes()
        self.shape = array.shape

    def array_view(self):
        return np.frombuffer(self.encoded, dtype=bool).reshape(self.shape)


class Tiny:
    def __init__(self, monkeypatch, plan=None, n=3):
        self.plan = plan or spec()
        self.state = cli.PilotState(self.plan, cli.WorkingMemory(cli.LIMITS["max_accounted_working_bytes"]))
        slides = tuple(Slide(i, i, str(100+i), "patient"+str(i), "A" if i < 2 else "B",
                             0 if i < 2 else 2, 2*i, 2*i+2, 2) for i in range(n))
        self.raws = {}
        for i, slide in enumerate(slides):
            data = np.zeros((2, 1536), np.float32)
            data[:, i % 4] = 1
            consumption = SimpleConsumption("feature:"+slide.slide_id, slide.slide_id, "synthetic",
                                            0, "0"*64, ("synthetic",))
            self.raws[i] = SimpleNamespace(slide=slide, raw=data, coordinates=np.array([[0, 0], [1, 1]]),
                                          consumption=consumption)
        masks = []
        for budget in (5, 100):
            selected = np.ones(n*2, bool)
            if budget == 100:
                selected[1::2] = False
            masks.append(SimpleNamespace(view=SimpleNamespace(held_out_fold=0, budget_percent=budget),
                backing=Backing(selected),
                counts=tuple(SimpleNamespace(retained_tiles=int(selected[2*i:2*i+2].sum()))
                             for i in range(n))))
        self.state.bundle = SimpleNamespace(slides=slides, masks=tuple(masks),
            feature_records=tuple(SimpleNamespace(slide_index=i, slide_id=s.slide_id, rows=2, size=0,
                consumer_asset=SimpleNamespace(asset_id="feature:"+s.slide_id)) for i, s in enumerate(slides)))
        self.state.guard = object()
        reader = SimpleNamespace(load_raw_slide=lambda guard, bundle, index: self.raws[index])
        monkeypatch.setattr(cli, "_reader", lambda: reader)

    def events(self, req=None):
        req = req or self.plan["pair_requests"][0]
        q = cli.prepare(self.state, self.raws[req["query_index"]], req)
        r = cli.prepare(self.state, self.raws[req["reference_index"]], req)
        observation = self.state.observation(req, req["reference_index"])
        return req, q, r, observation


@dataclass(frozen=True)
class SimpleConsumption:
    asset_id: str
    logical_name: str
    consumer_path: str
    producer_size: int
    producer_sha256: str
    semantic_checks: tuple


@pytest.mark.parametrize("name", cli.TOP_KEYS)
def test_missing_top_key(name):
    value = spec()
    del value[name]
    with pytest.raises(cli.PilotError):
        cli.validate_spec(value)


@pytest.mark.parametrize("field,value", [
    ("query_index", True), ("query_index", 0.0), ("query_fold", True),
    ("query_fold", 5), ("reference_index", -1), ("budget_percent", True),
    ("budget_percent", 5.0), ("budget_percent", 0), ("method", "ot"),
    ("selection", "tumor"), ("request_id", "../escape"), ("request_id", ""),
    ("request_id", "x"*129),
])
def test_request_types(field, value):
    item = request()
    item[field] = value
    with pytest.raises(cli.PilotError):
        cli.validate_spec(spec([item]))


@pytest.mark.parametrize("field", cli.LIMITS)
@pytest.mark.parametrize("kind", ("zero", "bool", "above"))
def test_limit_declarations(field, kind):
    value = spec()
    value["limits"][field] = {"zero": 0, "bool": True, "above": cli.LIMITS[field]+1}[kind]
    with pytest.raises(cli.PilotError):
        cli.validate_spec(value)


def test_duplicate_and_partner_seams():
    for items in ([request(), request()], [request(), request("different")],
                  [request(selection="all_true")]):
        with pytest.raises(cli.PilotError):
            cli.validate_spec(spec(items))
    filtered = row_request("filtered")
    with pytest.raises(cli.PilotError):
        cli.validate_spec(spec([], [filtered]))
    literal = row_request("literal", selection="literal_all_tissue")
    cli.validate_spec(spec([], [literal, filtered]))


def test_query_union_fold_and_extra_spec():
    for change in (lambda s: s.update(extra=1), lambda s: s.update(query_indices=[1]),
                   lambda s: s["pair_requests"].append(request("p2", q=1))):
        value = spec()
        change(value)
        if len(value["pair_requests"]) == 2:
            value["query_indices"] = [0, 1]
            value["pair_requests"][1]["query_fold"] = 1
        with pytest.raises(cli.PilotError):
            cli.validate_spec(value)


@pytest.mark.parametrize("selection", ("literal_all_tissue", "all_true", "filtered"))
def test_selection_layout_and_event_identity(monkeypatch, selection):
    req = request(selection=selection)
    tiny = Tiny(monkeypatch, spec([req]))
    tiny.raws[0].raw = np.asfortranarray(tiny.raws[0].raw)
    event = cli.prepare(tiny.state, tiny.raws[0], req)
    record = tiny.state.preparations[0]
    assert record["raw_f_contiguous"] and not record["raw_c_contiguous"]
    assert record["selected_f_contiguous"] == (selection == "literal_all_tissue")
    assert event.preparation_id == "p0000000000"
    cli.validate_preparation(tiny.state, event, req, 0)
    with pytest.raises(cli.PilotError):
        cli.validate_preparation(tiny.state, replace(event, array=event.array.copy()), req, 0)


@pytest.mark.parametrize("budget", (5, 100))
@pytest.mark.parametrize("method", cli.METHODS)
def test_native_tiny_pair(monkeypatch, budget, method):
    req = request(method=method, budget=budget)
    tiny = Tiny(monkeypatch, spec([req]))
    req, q, r, obs = tiny.events(req)
    value = cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=True)
    assert value >= 0 and obs["computed_valid"]
    assert obs["query_preparation_id"] == q.preparation_id
    assert obs["reference_preparation_id"] == r.preparation_id
    if method == "ot_cosine":
        assert obs["solver_result_code"] == 1 and obs["solver_warning_is_none"]
        assert obs["scalar_log_comparison"] == "equal"
        assert tiny.state.controls[req["request_id"], "scalar_log"]["comparison_count"] == 1
    else:
        assert obs["scalar_log_comparison"] == "not_applicable"


@pytest.mark.parametrize("value", (True, 1, np.float32(1), np.array([1.0]), complex(1), np.nan, np.inf))
def test_bad_distance_types(value):
    with pytest.raises(cli.PilotError):
        cli.double(value, "fixture")


def test_hex_signed_zero_and_adjacent_double():
    assert not cli.same_double(0.0, -0.0)
    values = [0.0, -0.0, 1.0, float(np.nextafter(1.0, 2.0))]
    for value in values:
        assert cli.same_double(value, float.fromhex(cli.csv_cell(value.hex())))


@pytest.mark.parametrize("code", (0, 2, True, np.bool_(True), 1.0, "1"))
def test_solver_status_failure(monkeypatch, code):
    tiny = Tiny(monkeypatch)
    req, q, r, obs = tiny.events()
    original = cli.ot.emd2

    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        result[1]["result_code"] = code
        return result

    monkeypatch.setattr(cli.ot, "emd2", altered)
    # Preserve the declared signature, so this tests the returned status.
    altered.__signature__ = inspect.signature(original)
    with pytest.raises(cli.PilotError):
        cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=False)
    assert not obs["computed_valid"]


@pytest.mark.parametrize("mutation", ("extra", "missing", "warning", "dual_nan", "dual_shape", "fp32"))
def test_log_interface_failures(monkeypatch, mutation):
    tiny = Tiny(monkeypatch)
    req, q, r, obs = tiny.events()
    original = cli.ot.emd2

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        log = result[1]
        if mutation == "extra":
            log["G"] = np.ones((2, 2))
        elif mutation == "missing":
            del log["u"]
        elif mutation == "warning":
            log["warning"] = "not optimal"
        elif mutation == "dual_nan":
            log["u"][0] = np.nan
        elif mutation == "dual_shape":
            log["v"] = np.zeros(3)
        else:
            result[0] = np.float32(result[0])
        return result

    changed.__signature__ = inspect.signature(original)
    monkeypatch.setattr(cli.ot, "emd2", changed)
    with pytest.raises(cli.PilotError):
        cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=False)


def test_warning_ignore_and_errstate_restored(monkeypatch):
    tiny = Tiny(monkeypatch)
    req, q, r, obs = tiny.events()
    original = cli.ot.dist
    before = np.geterr()

    def warned(*args, **kwargs):
        warnings.warn("synthetic warning", RuntimeWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(cli.ot, "dist", warned)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filters = list(warnings.filters)
        with np.errstate(all="raise"):
            with pytest.raises(cli.PilotError, match="captured"):
                cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=False)
            assert np.geterr() == dict(divide="raise", invalid="raise", over="raise", under="raise")
        assert warnings.filters == filters
    assert np.geterr() == before and obs["captured_warning_count"] == 1
    assert tiny.state.warnings[0]["request_id"] == req["request_id"]


def test_original_baseexception_keeps_warning_notes(monkeypatch):
    tiny = Tiny(monkeypatch)
    req, q, r, obs = tiny.events()
    error = KeyboardInterrupt("original")

    def interrupted(*args, **kwargs):
        warnings.warn("before interrupt", UserWarning)
        raise error

    monkeypatch.setattr(cli.ot, "dist", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=False)
    assert caught.value is error and any("before interrupt" in n for n in error.__notes__)


def test_signed_costs_are_held_without_clipping(monkeypatch):
    tiny = Tiny(monkeypatch)
    req, q, r, obs = tiny.events()
    monkeypatch.setattr(cli.ot, "dist", lambda a, b, **kw: np.full((len(a), len(b)), -0.125, np.float32))
    with pytest.raises(cli.PilotPromotionHeld, match="signed"):
        cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=False)
    assert float.fromhex(obs["cosine_min_hex"]) == -0.125
    assert float.fromhex(obs["distance_hex"]) == -0.125 and not obs["computed_valid"]


def test_pair_cap_is_before_cost_allocation(monkeypatch):
    tiny = Tiny(monkeypatch)
    req, q, r, obs = tiny.events()
    tiny.state.spec["limits"]["max_pair_cost_cells"] = 3
    monkeypatch.setattr(cli.ot, "dist", lambda *a, **k: pytest.fail("cost allocation reached"))
    with pytest.raises(cli.PilotError, match="cell"):
        cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=False)


def test_working_memory_restoration_and_preallocation():
    budget = cli.WorkingMemory(32)
    budget.hold("retained", 16)
    with pytest.raises(cli.PilotError):
        with budget.temporary(17):
            pytest.fail("allocation boundary reached")
    with pytest.raises(KeyboardInterrupt):
        with budget.temporary(16):
            raise KeyboardInterrupt
    assert budget.live == 16 and budget.peak == 32
    assert budget.summary()["basis"].endswith("not_rss")


def test_bob_half_integer_and_empty_library_source_cases(monkeypatch):
    req = request(method="yottixel")
    tiny = Tiny(monkeypatch, spec([req]))
    req, q, r, obs = tiny.events(req)
    # A legitimate even query median can be a half integer.
    a = np.zeros((2, 1535), np.uint8)
    a[1, 0] = 1
    b = np.zeros((1, 1535), np.uint8)
    q = cli.Prepared(q.preparation_id, 0, q.recipe, a, 2, q.reservation)
    r = cli.Prepared(r.preparation_id, 1, r.recipe, b, 1, r.reservation)
    tiny.state.prepared_events[q.preparation_id] = q
    tiny.state.prepared_events[r.preparation_id] = r
    assert cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=False) == 0.5


def baseline_fixture(tiny):
    rows = [{"Tumor.group": s.diagnostic_group, "Slide.Scan.File": s.slide_id,
             "Sample.ID": "sample"+str(s.slide_index), "UID.Subject": s.patient_id}
            for s in tiny.state.bundle.slides]
    metadata = cli.cpu.csv_bytes(tuple(rows[0]), rows)
    descriptor = lambda data, path: {"path": path, **cli.binding(data)}
    methods = [m for m in cli.METHODS if any(
        r["method"] == m for r in tiny.plan["pair_requests"]+tiny.plan["row_requests"])]
    sources = {"metadata": descriptor(metadata, "/historical/metadata.csv")}
    matrices = {}
    for method in methods:
        values = np.zeros((len(rows), len(rows)), np.float64)
        for i in range(len(rows)):
            for j in range(len(rows)):
                if method == "ot_cosine":
                    values[i, j] = float(i != j)
                else:
                    values[i, j] = cli.bob.hamming_min_median(
                        cli.bob.features_to_barcodes(tiny.raws[i].raw),
                        cli.bob.features_to_barcodes(tiny.raws[j].raw))
        matrix = pd.DataFrame(values, index=[r["Slide.Scan.File"] for r in rows],
                              columns=[r["Slide.Scan.File"] for r in rows])
        data = pickle.dumps(matrix, protocol=4)
        sources["matrix_"+method] = descriptor(data, "/historical/"+method+".pkl")
        matrices[method] = data
    producer_spec = cli.cpu.json_bytes({"schema_version": 1, "task": "hoptimus_main",
        "run_id": "synthetic_baseline", "expected": {"raw_slides": len(rows)}, "sources": sources})
    records = [{"kind": "spec", "name": "spec", **descriptor(producer_spec, "/historical/spec.json")}]
    records += [{"kind": "source", "name": name, **value} for name, value in sources.items()]
    inputs = cli.cpu.json_bytes({"schema_version": 1, "task": "hoptimus_main",
        "run_id": "synthetic_baseline", "spec_sha256": cli.binding(producer_spec)["sha256"],
        "bindings": records})
    registry = {name: {"size": 0, "sha256": "0"*64} for name in cli.MAIN_OUTPUTS}
    registry["input_bindings.json"] = cli.binding(inputs)
    complete = cli.cpu.json_bytes({"schema_version": 1, "task": "hoptimus_main",
        "scope": "hoptimus_main_saved_rank_evaluation_only", "run_id": "synthetic_baseline",
        "spec_sha256": cli.binding(producer_spec)["sha256"], "outputs": registry})
    review = cli.cpu.json_bytes({"schema_version": 1, "disposition": "accepted",
        "scope": "Actual main saved-source evaluation and filter/perturbation preparation bundles only",
        "branches": {"main": {"disposition": "accepted"}},
        "artifacts": {"historical-key-is-data": cli.binding(complete)["sha256"]}})
    data = {"review": review, "complete": complete, "spec": producer_spec,
            "input_bindings": inputs, "metadata": metadata}
    declaration = {}
    for name, value in data.items():
        suffix = "review57" if name == "review" else name
        declaration[name] = {"asset_id": "baseline:"+suffix, "role": "baseline_"+name,
                             **descriptor(value, "/current/"+name)}
    declaration["matrices"] = {method: {"asset_id": "baseline:matrix:"+method,
        "role": "baseline_matrix", **descriptor(value, "/current/"+method)}
        for method, value in matrices.items()}
    declaration["historical_review_complete_key"] = "historical-key-is-data"
    expected = {name+"_sha256": cli.binding(value)["sha256"] for name, value in data.items()
                if name != "metadata"}
    expected["run_id"] = "synthetic_baseline"
    tiny.plan["baseline_assets"], tiny.plan["baseline_expected"] = declaration, expected
    payloads = {declaration[name]["asset_id"]: value for name, value in data.items()}
    payloads.update({declaration["matrices"][m]["asset_id"]: value for m, value in matrices.items()})
    tiny.state.guard = SimpleNamespace(read_verified=lambda aid: payloads[aid])
    tiny.state.baseline = cli.baseline_controls(tiny.state.guard, tiny.plan, tiny.state.bundle)
    return payloads


@pytest.mark.parametrize("method", cli.METHODS)
def test_full_rows_and_both_budget_actual_event_links(monkeypatch, method):
    rows = [row_request("literal", method, "literal_all_tissue"),
            row_request("primary", method), row_request("sensitivity", method, budget=100)]
    tiny = Tiny(monkeypatch, spec([], rows))
    baseline_fixture(tiny)
    seen = []
    original = cli.compute_pair

    def observed(state, req, q, r, cell, **kwargs):
        seen.append((req["request_id"], cell["reference_index"], q, r))
        return original(state, req, q, r, cell, **kwargs)

    monkeypatch.setattr(cli, "compute_pair", observed)
    cli.complete_rows(tiny.state)
    for name, ri, q, r in seen:
        cell = tiny.state.row_cells[name][ri]
        assert cell["query_preparation_id"] == q.preparation_id
        assert cell["reference_preparation_id"] == r.preparation_id
    primary_queries = {q.preparation_id for name, ri, q, r in seen if name == "primary"}
    sensitivity_queries = {q.preparation_id for name, ri, q, r in seen if name == "sensitivity"}
    assert len(primary_queries) == len(sensitivity_queries) == 1
    assert primary_queries.isdisjoint(sensitivity_queries)
    assert all(tiny.state.results[r["request_id"]]["status"] == "returned" for r in rows)
    assert all(len(tiny.state.row_cells[r["request_id"]]) == 3 for r in rows)
    assert tiny.state.results["primary"]["metrics"][0]["returned_depth"] == 2
    assert tiny.state.results["primary"]["metrics"][0]["short_list"]
    assert tiny.state.memory.live >= 0


@pytest.mark.parametrize("empty", ("query", "library"))
def test_empty_rows_normal_diagnostics_after_controls(monkeypatch, empty):
    rows = [row_request("literal", "yottixel", "literal_all_tissue"),
            row_request("filtered", "yottixel")]
    tiny = Tiny(monkeypatch, spec([], rows))
    baseline_fixture(tiny)
    selected = np.ones(6, bool)
    selected[:2] = empty != "query"
    if empty == "library":
        selected[2:] = False
    loaded = tiny.state.bundle.masks[0]
    loaded.backing = Backing(selected)
    loaded.counts = tuple(SimpleNamespace(retained_tiles=int(selected[i:i+2].sum()))
                          for i in range(0, 6, 2))
    cli.complete_rows(tiny.state)
    result = tiny.state.results["filtered"]
    assert result["status"] == ("empty_query" if empty == "query" else "empty_reference_library")
    assert result["controls_passed"] and result["metrics"][0]["hit_yield"] == 0
    assert result["metrics"][0]["conditional_precision"] is None


def test_full_row_control_difference_holds_not_returned(monkeypatch):
    rows = [row_request("literal", "yottixel", "literal_all_tissue"),
            row_request("filtered", "yottixel")]
    tiny = Tiny(monkeypatch, spec([], rows))
    payloads = baseline_fixture(tiny)
    matrix = pickle.loads(payloads["baseline:matrix:yottixel"])
    matrix.iat[0, 1] += 0.5
    payloads["baseline:matrix:yottixel"] = pickle.dumps(matrix, protocol=4)
    with pytest.raises(cli.PilotPromotionHeld):
        cli.complete_rows(tiny.state)
    assert not tiny.state.row_cells["literal"][1]["computed_valid"]
    assert tiny.state.results["literal"]["status"] == "held"
    assert tiny.state.results["filtered"]["status"] != "returned"


@pytest.mark.parametrize("axis", ("bool", "duplicates", "missing"))
def test_complete_baseline_axis_guards(monkeypatch, axis):
    rows = [row_request("literal", "yottixel", "literal_all_tissue")]
    tiny = Tiny(monkeypatch, spec([], rows))
    payloads = baseline_fixture(tiny)
    data = pickle.loads(payloads["baseline:matrix:yottixel"])
    data.index = {"bool": [True, "101", "102"],
                  "duplicates": [100, "100", "102"], "missing": ["100", "101", "999"]}[axis]
    payloads["baseline:matrix:yottixel"] = pickle.dumps(data, protocol=4)
    with pytest.raises(Exception):
        cli.load_baseline(tiny.state, "yottixel")


def test_scalar_default_one_checked_even_with_explicit_log_threads(monkeypatch):
    original = inspect.signature(cli.ot.emd2)
    changed = original.replace(parameters=[
        p.replace(default=2) if p.name == "numThreads" else p
        for p in original.parameters.values()])
    monkeypatch.setattr(cli.ot.emd2, "__signature__", changed, raising=False)
    with pytest.raises(cli.PilotError, match="default"):
        cli.pot_interface()


def test_example_is_strict_unbound_synthetic_only():
    path = Path(__file__).parents[1]/"examples/r2_1_hoptimus_filter_retrieval.json"
    value = cli.cpu.strict_json(path.read_bytes())
    cli.validate_spec(value)
    assert value["run_id"] == "SYNTHETIC_ONLY_UNBOUND_EXAMPLE"
    assert value["code_files"] == {} and not value["runtime_assets"]["implementations"]


def test_runtime_alias_pool_order_and_membership(monkeypatch):
    base = cli.fit._runtime_profile()
    paths = {str(Path(p).resolve()) for p in (
        cli.sys.executable, cli.pd.__file__, cli.threadpoolctl.__file__, cli.ot.__file__,
        inspect.getfile(cli.ot.dist), cli.simplex.__file__, cli.emd_wrap.__file__,
        cli.np._core._multiarray_umath.__file__)}
    paths.update(str(Path(v["path"]).resolve()) for v in base["implementation"].values())
    paths.update(str(Path(v["filepath"]).resolve()) for v in base["native_pools"])
    declarations = [{"asset_id": f"runtime:impl:synthetic{i}", "role": "runtime_implementation",
        "path": path, "size": 0, "sha256": "0"*64} for i, path in enumerate(sorted(paths))]
    original = cli.runtime_snapshot(declarations)
    copied = json.loads(cli.cpu.json_bytes(base))
    copied["native_pools"].reverse()
    monkeypatch.setattr(cli.fit, "_runtime_profile", lambda: copied)
    reordered = cli.runtime_snapshot(declarations)
    assert original["profile"] == reordered["profile"]
    assert original["profile"]["pot_interface"]["numeric_dtype_recipe"]["status"] == "declared"
    with pytest.raises(cli.PilotError, match="missing/extra"):
        cli.runtime_snapshot(declarations[:-1])
    pool = copied["native_pools"][0]
    pool["version"] = "synthetic-changed"
    assert cli.runtime_snapshot(declarations)["profile"] != original["profile"]


def test_baseline_control_receipt_and_alias_negatives(monkeypatch):
    rows = [row_request("literal", selection="literal_all_tissue")]
    tiny = Tiny(monkeypatch, spec([], rows))
    payloads = baseline_fixture(tiny)
    assert tiny.state.baseline["sources"]["metadata"]["path"] == "/historical/metadata.csv"
    original = payloads["baseline:review57"]
    review = json.loads(original)
    review["artifacts"]["historical-key-is-data"] = {"sha256": "0"*64, "size": 1}
    payloads["baseline:review57"] = cli.cpu.json_bytes(review)
    tiny.plan["baseline_expected"]["review_sha256"] = cli.binding(payloads["baseline:review57"])["sha256"]
    with pytest.raises(cli.PilotError, match="historical"):
        cli.baseline_controls(tiny.state.guard, tiny.plan, tiny.state.bundle)


def test_empty_explicit_probe_never_calls_pair(monkeypatch):
    tiny = Tiny(monkeypatch)
    loaded = tiny.state.bundle.masks[0]
    loaded.backing = Backing(np.zeros(6, bool))
    loaded.counts = tuple(SimpleNamespace(retained_tiles=0) for _ in range(3))
    monkeypatch.setattr(cli, "compute_pair", lambda *a, **k: pytest.fail("empty pair computed"))
    cli.explicit_pairs(tiny.state)
    assert not tiny.state.pair_probes[0]["computed_valid"]
    assert tiny.state.pair_probes[0]["distance_hex"] is None
    assert tiny.state.pair_probes[0]["query_preparation_id"] is None
    assert tiny.state.results["pair"]["status"] == "structural_skip"


def test_reference_preparations_not_a_resident_library(monkeypatch):
    import gc
    rows = [row_request("literal", "yottixel", "literal_all_tissue"),
            row_request("filtered", "yottixel")]
    tiny = Tiny(monkeypatch, spec([], rows))
    baseline_fixture(tiny)
    cli.complete_rows(tiny.state)
    gc.collect()
    events = list(tiny.state.prepared_events.values())
    assert all(e.slide_index in tiny.plan["query_indices"] for e in events)


@pytest.mark.parametrize("fault", ("zero", "nonfinite"))
def test_invalid_norm_or_feature_fails_without_removal(monkeypatch, fault):
    tiny = Tiny(monkeypatch)
    if fault == "zero":
        tiny.raws[0].raw[:] = 0
    else:
        tiny.raws[0].raw[0, 0] = np.nan
    with pytest.raises(cli.PilotError):
        cli.prepare(tiny.state, tiny.raws[0], request())
    assert tiny.raws[0].raw.shape == (2, 1536)


def test_owned_foreign_marker_recovery_refusal(tmp_path, monkeypatch):
    root = tmp_path/"out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    output.close()
    (output.out/"COMPLETE.json").write_text("foreign")
    error = KeyboardInterrupt("primary")
    with pytest.raises(Exception, match="foreign"):
        output.fail(error, [])
    assert (output.out/"COMPLETE.json").read_text() == "foreign"
    assert not (output.out/"FAILED").exists()


def test_replaced_directory_recovery_refusal(tmp_path, monkeypatch):
    root = tmp_path/"out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    output.close()
    output.out.rename(root/"retained-owned")
    output.out.mkdir()
    with pytest.raises(Exception, match="identity"):
        output.fail(OSError("primary"), [])
    assert not (output.out/"FAILED").exists()


@pytest.mark.parametrize("missing", ("spec", "spec-sha256", "out", "out-root"))
def test_public_parser_requires_each_argument(missing):
    argv = []
    for option in ("spec", "spec-sha256", "out", "out-root"):
        if option != missing:
            argv += ["--"+option, "SYNTHETIC_ONLY"]
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 2


def test_writer_extra_artifact_not_accepted(tmp_path, monkeypatch):
    root = tmp_path/"out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    bindings = {}
    try:
        cli.json_output(output, "declared.json", {"synthetic": True}, bindings,
                        cli.WorkingMemory(64*cli.MIB))
        (output.out/"foreign.json").write_text("{}")
        with pytest.raises(Exception, match="allowlist"):
            output.check(bindings)
    finally:
        output.close()


def test_csv_record_and_type_limits_before_success(tmp_path, monkeypatch):
    root = tmp_path/"out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    try:
        with pytest.raises(cli.PilotError, match="record ceiling"):
            cli.csv_output(output, "large.csv", ("field",), [{"field": "x"*8449}],
                           {}, cli.WorkingMemory(64*cli.MIB), 1)
        with pytest.raises(cli.PilotError):
            cli.csv_cell(1.25)
        assert not (output.out/"COMPLETE.json").exists()
    finally:
        output.close()


@pytest.mark.parametrize("filtered_between", (False, True))
def test_fil1_literal_partner_survives_filtered_structural_skip(monkeypatch, filtered_between):
    literal = request("literal", "yottixel", "literal_all_tissue")
    filtered = request("filtered", "yottixel")
    all_true = request("alltrue", "yottixel", "all_true")
    pairs = [literal, filtered, all_true] if filtered_between else [literal, all_true, filtered]
    tiny = Tiny(monkeypatch, spec(pairs), n=2)
    baseline_fixture(tiny)
    loaded = tiny.state.bundle.masks[0]
    loaded.backing = Backing(np.array([False, False, True, True]))
    loaded.counts = (SimpleNamespace(retained_tiles=0), SimpleNamespace(retained_tiles=2))
    cli.explicit_pairs(tiny.state)
    assert tiny.state.results["literal"]["status"] == "valid_distance"
    assert tiny.state.results["filtered"]["status"] == "structural_skip"
    assert tiny.state.results["alltrue"]["status"] == "valid_distance"
    assert tiny.state.controls["alltrue", "literal_all_true"]["status"] == "equal"


@pytest.mark.parametrize("identifier", ("0xprobe", "-0xprobe"))
def test_fil2_hex_prefixed_identifier_serializes_as_text(tmp_path, monkeypatch, identifier):
    req = request(identifier, "yottixel")
    tiny = Tiny(monkeypatch, spec([req]))
    cli.explicit_pairs(tiny.state)
    root = tmp_path/"synthetic-out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    try:
        bindings = {}
        cli.csv_output(output, "pair_probes.csv", cli.PAIR_HEADER, tiny.state.pair_probes,
                       bindings, tiny.state.memory, 1)
        parsed = list(cli.producer.csv_rows(
            io.BytesIO((output.out/"pair_probes.csv").read_bytes()), cli.PAIR_HEADER, "fixture",
            compressed=False, expanded_limit=64*cli.MIB, row_limit=1, record_limit=8448))
        assert parsed[0]["request_id"] == identifier
        assert cli.same_double(float.fromhex(parsed[0]["distance_hex"]),
                               float.fromhex(tiny.state.pair_probes[0]["distance_hex"]))
    finally:
        output.close()


def test_fil2_invalid_numeric_hex_column_still_rejected(tmp_path, monkeypatch):
    root = tmp_path/"synthetic-out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    try:
        with pytest.raises((ValueError, cli.PilotError)):
            cli.csv_output(output, "numeric.csv", ("distance_hex",),
                           [{"distance_hex": "0xprobe"}], {}, cli.WorkingMemory(64*cli.MIB), 1)
    finally:
        output.close()


def test_fil3_underflowed_raw_norm_preserves_frozen_floor_preprocessing(monkeypatch):
    tiny = Tiny(monkeypatch)
    req = request()
    for index in (0, 1):
        tiny.raws[index].raw[:] = 0
        tiny.raws[index].raw[:, index] = np.float32(1e-24)
    assert (np.linalg.norm(tiny.raws[0].raw, axis=1) == 0).all()
    original = [tiny.state.generator.process_single_slide(tiny.raws[i].raw, str(i))["features"]
                for i in (0, 1)]
    assert (np.linalg.norm(original[0], axis=1) > 0).all()
    assert float(tiny.state.generator.compute_ot_distance(*original)) == 1.0
    req, query, reference, observation = tiny.events(req)
    assert np.array_equal(query.array, original[0])
    assert np.array_equal(reference.array, original[1])
    assert cli.compute_pair(tiny.state, req, query, reference, observation,
                            scalar_control=True) == 1.0
    assert observation["computed_valid"]


@pytest.mark.parametrize("phase,side,fault", [
    ("pair", "reference", "norm"), ("row", "query", "norm"),
    ("pair", "query", "eio"), ("row", "reference", "eio"),
    ("pair", "reference", "eio"), ("row", "reference", "norm"),
    ("pair", "query", "interrupt"), ("row", "reference", "interrupt"),
])
def test_fil4_active_cell_acquisition_and_preparation_failure(monkeypatch, phase, side, fault):
    req = (request("failure", selection="literal_all_tissue") if phase == "pair"
           else row_request("failure", "ot_cosine", "literal_all_tissue"))
    tiny = Tiny(monkeypatch, spec([req], []) if phase == "pair" else spec([], [req]))
    baseline_fixture(tiny)
    index = 0 if side == "query" else 1
    error = KeyboardInterrupt("synthetic primary") if fault == "interrupt" else OSError(5, "synthetic EIO")
    if fault == "norm":
        tiny.raws[index].raw[:] = 0
    else:
        original = tiny.state.raw

        def failed_load(current, *, query):
            if current == index:
                raise error
            return original(current, query=query)

        monkeypatch.setattr(tiny.state, "raw", failed_load)
    with pytest.raises(BaseException) as caught:
        (cli.explicit_pairs if phase == "pair" else cli.complete_rows)(tiny.state)
    if fault != "norm":
        assert caught.value is error
    cells = tiny.state.pair_probes if phase == "pair" else tiny.state.row_cells["failure"]
    active = cells[-1]
    assert not cli._skip(active)
    assert active["observation_status"] == "failed" and not active["computed_valid"]
    assert tiny.state.results["failure"]["status"] == "failed"
    assert active["reference_preparation_id"] is None
    if side == "query":
        assert active["query_preparation_id"] is None
    else:
        assert active["query_preparation_id"] is not None


def test_fil5_reference_array_lifetime_at_next_raw_load(monkeypatch):
    import gc
    import weakref
    rows = [row_request("literal", "yottixel", "literal_all_tissue"),
            row_request("primary", "yottixel"), row_request("sensitivity", "yottixel", budget=100)]
    tiny = Tiny(monkeypatch, spec([], rows), n=4)
    baseline_fixture(tiny)
    tracked, boundaries = {}, []
    original_prepare, original_raw = cli.prepare, tiny.state.raw

    def tracked_prepare(state, raw_slide, req):
        event = original_prepare(state, raw_slide, req)
        if event.array is not None:
            tracked[event.preparation_id] = (
                weakref.ref(event.array), event.reservation, event.slide_index)
        return event

    def at_next_load(index, *, query):
        gc.collect()
        unreserved = [(pid, slide) for pid, (ref, token, slide) in tracked.items()
                      if slide not in tiny.plan["query_indices"] and ref() is not None
                      and token not in tiny.state.memory.retained]
        boundaries.append((index, unreserved))
        return original_raw(index, query=query)

    monkeypatch.setattr(cli, "prepare", tracked_prepare)
    monkeypatch.setattr(tiny.state, "raw", at_next_load)
    cli.complete_rows(tiny.state)
    assert len(boundaries) >= 4
    assert all(not live for _, live in boundaries), boundaries
    assert tiny.state.query_prepared
    assert all(event.reservation in tiny.state.memory.retained
               for event in tiny.state.query_prepared.values())


@pytest.mark.parametrize("fault,marker", [
    ("warning", "FIL6_RETURN_ONLY_WARNING"),
    ("boolean_status", "True"),
    ("string_status", "FIL6_BAD_STATUS"),
])
def test_fil6_returned_solver_evidence_survives_existing_diagnostic_fields(monkeypatch, fault, marker):
    tiny = Tiny(monkeypatch)
    original = cli.ot.emd2

    def altered(*args, **kwargs):
        value, log = original(*args, **kwargs)
        log = dict(log)
        if fault == "warning":
            log["warning"] = marker
        else:
            log["result_code"] = True if fault == "boolean_status" else marker
        return value, log

    altered.__signature__ = inspect.signature(original)
    monkeypatch.setattr(cli.ot, "emd2", altered)
    with pytest.raises(cli.PilotError) as caught:
        cli.explicit_pairs(tiny.state)
    projected = {"error": {"type": type(caught.value).__name__, "message": str(caught.value),
                          "notes": list(getattr(caught.value, "__notes__", ()))},
                 "warnings": tiny.state.warnings, "reason_codes": [str(caught.value)]}
    encoded = cli.cpu.json_bytes(projected)
    assert marker.encode() in encoded
    assert b"request_id" in encoded and b"result_code" in encoded and b"warning" in encoded
    assert tiny.state.warnings == []
    assert not tiny.state.pair_probes[-1]["computed_valid"]


def write_fil6_diagnostic(tmp_path, monkeypatch, error):
    projected = {"error": {"type": type(error).__name__, "message": str(error),
                          "notes": list(getattr(error, "__notes__", ()))},
                 "warnings": [], "reason_codes": [str(error)]}
    root = tmp_path/"synthetic-diagnostic"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    try:
        bindings = {}
        cli.json_output(output, "DIAGNOSTIC.json", projected, bindings,
                        cli.WorkingMemory(64*cli.MIB))
        raw = (output.out/"DIAGNOSTIC.json").read_bytes()
        assert json.loads(raw) == projected
        assert bindings["DIAGNOSTIC.json"]["sha256"] == hashlib.sha256(raw).hexdigest()
    finally:
        output.close()
    return projected


@pytest.mark.parametrize("fault", ("huge_integer", "shared_container"))
def test_fil6_residual_native_fault_has_bounded_persisted_context(tmp_path, monkeypatch, fault):
    tiny = Tiny(monkeypatch)
    original = cli.ot.emd2
    digit_limit = sys.get_int_max_str_digits()
    value = 10**5000
    if fault == "shared_container":
        value = "x"*1000
        for _ in range(5):
            value = [value]*6

    def altered(*args, **kwargs):
        distance, log = original(*args, **kwargs)
        log = dict(log, result_code=value)
        if fault == "shared_container":
            log["warning"] = value
        return distance, log

    altered.__signature__ = inspect.signature(original)
    monkeypatch.setattr(cli.ot, "emd2", altered)
    with pytest.raises(cli.PilotError) as caught:
        cli.explicit_pairs(tiny.state)
    projected = write_fil6_diagnostic(tmp_path, monkeypatch, caught.value)
    notes = projected["error"]["notes"]
    assert notes
    assert all(len(json.dumps(note, ensure_ascii=True).encode("utf8")) <= 4096 for note in notes)
    note = notes[0]
    assert "pair" in note and "result_code" in note and "warning" in note
    assert "omitted" in note
    assert ("bit_length" in note) if fault == "huge_integer" else ("builtins.list" in note)
    assert sys.get_int_max_str_digits() == digit_limit
    assert tiny.state.warnings == []
    assert tiny.state.pair_probes[-1]["observation_status"] == "failed"
    assert not tiny.state.pair_probes[-1]["computed_valid"]
    assert "solver_result_code_raw" not in tiny.state.pair_probes[-1]


@pytest.mark.parametrize("text", [
    "\U0001f600"*256,
    "\x00\x01\b\f\n\r\t\\\""*256,
    "\ud800"*256,
    "quoted \\\" Unicode \U0001f600 control \x00 "*256,
])
def test_fil6_complete_note_json_string_budget(tmp_path, monkeypatch, text):
    error = cli.PilotError("synthetic primary")
    cli._returned_solver_note(error, request("escaped-probe"), 1,
                              {"result_code": text, "warning": text})
    projected = write_fil6_diagnostic(tmp_path, monkeypatch, error)
    assert len(projected["error"]["notes"]) == 1
    note = projected["error"]["notes"][0]
    encoded = json.dumps(note, ensure_ascii=True).encode("utf8")
    assert len(encoded) <= 4096
    assert json.loads(encoded) == note
    assert "escaped-probe" in note and "builtins.str" in note
    assert "truncated" in note or "omitted" in note
    if text == "\U0001f600"*256:
        assert "4096-byte" in note and len(encoded) >= 4080


class Fil6NoRepresentation:
    calls = 0

    def __repr__(self):
        self.calls += 1
        raise AssertionError("unsupported representation must not be called")

    __str__ = __repr__


@pytest.mark.parametrize("kind", ("object", "list", "dict", "set", "cycle", "int_subclass"))
def test_fil6_unsupported_types_never_expand_or_invoke_representation(kind):
    class UnsupportedInt(int):
        calls = 0
        __repr__ = Fil6NoRepresentation.__repr__
        __str__ = Fil6NoRepresentation.__str__

    sentinel = Fil6NoRepresentation()
    value = {"object": sentinel, "list": [sentinel],
             "dict": {"bad": sentinel}, "set": {sentinel},
             "cycle": [], "int_subclass": UnsupportedInt(1)}[kind]
    if kind == "cycle":
        value.append(value)
    error = cli.PilotError("synthetic primary")
    cli._returned_solver_note(error, request("unsupported-probe"), 1,
                              {"result_code": value, "warning": value})
    notes = error.__notes__
    assert len(notes) == 1
    assert len(json.dumps(notes[0], ensure_ascii=True).encode("utf8")) <= 4096
    assert "unsupported-probe" in notes[0] and "omitted" in notes[0]
    assert type(value).__name__ in notes[0]
    assert sentinel.calls == 0
    if kind == "int_subclass":
        assert value.calls == 0


@pytest.mark.parametrize("value,marker", [
    (None, "None"), (True, "True"), (False, "False"), (1, "1"),
    (np.int64(1), "1"), (np.uint64(2**64-1), "18446744073709551615"),
    ("ordinary warning", "ordinary warning"), (-0.0, (-0.0).hex()),
    (float.fromhex("0x1.0000000000001p+0"), "0x1.0000000000001p+0"),
    (np.float32(0.5), float(np.float32(0.5)).hex()),
    (float("nan"), "nan"), (float("inf"), "inf"),
])
def test_fil6_supported_scalar_summaries(value, marker):
    error = cli.PilotError("synthetic primary")
    cli._returned_solver_note(error, request("scalar-probe"), 1,
                              {"result_code": value, "warning": value})
    note = error.__notes__[0]
    assert "scalar-probe" in note and marker in note
    assert type(value).__name__ in note
    assert len(json.dumps(note, ensure_ascii=True).encode("utf8")) <= 4096


@pytest.mark.parametrize("fault", ("exception", "interrupt", "non_ascii"))
def test_fil6_formatting_failure_preserves_primary_and_bounded_type_context(
        tmp_path, monkeypatch, fault):
    class FormattingFailure(RuntimeError):
        __str__ = Fil6NoRepresentation.__str__
        __repr__ = Fil6NoRepresentation.__repr__

    primary = cli.PilotError("original validation failure")
    with monkeypatch.context() as patch:
        def unavailable(value):
            if fault == "non_ascii":
                return "\U0001f600".encode("utf8")
            raise FormattingFailure() if fault == "exception" else KeyboardInterrupt()

        patch.setattr(cli.cpu, "json_bytes", unavailable)
        cli._returned_solver_note(primary, request("formatting-probe"), 1,
                                  {"result_code": 10**5000, "warning": Fil6NoRepresentation()})
    assert type(primary) is cli.PilotError and str(primary) == "original validation failure"
    projected = write_fil6_diagnostic(tmp_path, monkeypatch, primary)
    assert len(projected["error"]["notes"]) == 1
    note = projected["error"]["notes"][0]
    assert "formatting-probe" in note and "formatting unavailable" in note
    assert "builtins.int" in note and "Fil6NoRepresentation" in note
    assert len(json.dumps(note, ensure_ascii=True).encode("utf8")) <= 4096


def test_fil6_worst_escaped_context_precedes_truncated_detail():
    kind = type("\U0001f600"*1000, (), {"__module__": "\U0001f600"*1000})
    value = kind()
    req = request("\U0001f600"*1000)
    error = cli.PilotError("synthetic primary")
    cli._returned_solver_note(error, req, 1, {"result_code": value, "warning": value})
    note = error.__notes__[0]
    prefix = "Bounded returned OT interface evidence: "
    context = json.loads(note[len(prefix):].split("; details:", 1)[0])
    assert set(context) == {"request_id", "result_code_type", "warning_type"}
    assert all("\U0001f600" in text and "omitted" in text for text in context.values())
    assert len(json.dumps(note, ensure_ascii=True).encode("utf8")) <= 4096


def test_fil6_unsupported_metaclass_cannot_trigger_scalar_conversion():
    class NoEquality(type):
        comparisons = 0

        def __eq__(cls, other):
            cls.comparisons += 1
            raise AssertionError("unsupported type equality must not be called")

    kind = NoEquality("NoScalarConversion", (Fil6NoRepresentation,), {})
    value = kind()
    error = cli.PilotError("synthetic primary")
    cli._returned_solver_note(error, request("type-identity-probe"), 1,
                              {"result_code": value, "warning": value})
    note = error.__notes__[0]
    assert "unsupported container/type" in note and "NoScalarConversion" in note
    assert kind.comparisons == value.calls == 0
    assert len(json.dumps(note, ensure_ascii=True).encode("utf8")) <= 4096


def test_fil6_native_validation_primary_survives_formatter_failure(monkeypatch):
    tiny = Tiny(monkeypatch)
    req, query, reference, observation = tiny.events()
    original_solve, original_note = cli.ot.emd2, cli._returned_solver_note
    original_json = cli.cpu.json_bytes
    primaries = []

    def altered(*args, **kwargs):
        distance, log = original_solve(*args, **kwargs)
        return distance, dict(log, result_code=True)

    def capture_primary(error, *args):
        primaries.append(error)
        original_note(error, *args)

    def unavailable(value):
        if type(value) is dict and "request_id" in value:
            raise OSError("synthetic formatter unavailable")
        return original_json(value)

    altered.__signature__ = inspect.signature(original_solve)
    monkeypatch.setattr(cli.ot, "emd2", altered)
    monkeypatch.setattr(cli, "_returned_solver_note", capture_primary)
    monkeypatch.setattr(cli.cpu, "json_bytes", unavailable)
    with pytest.raises(cli.PilotError, match="nonintegral/Boolean") as caught:
        cli.compute_pair(tiny.state, req, query, reference, observation, scalar_control=False)
    assert primaries == [caught.value]
    assert "formatting unavailable" in caught.value.__notes__[0]
    assert "builtins.bool" in caught.value.__notes__[0]
    assert not observation["computed_valid"]


def fil6_c4_native_cell(monkeypatch, phase, status, *, malformed_log=None, formatting_failure=False):
    req = (request("status-export") if phase == "pair" else
           row_request("status-export", "ot_cosine", "literal_all_tissue"))
    tiny = Tiny(monkeypatch, spec([req], []) if phase == "pair" else spec([], [req]))
    if phase == "row":
        baseline_fixture(tiny)
    original_solve, original_note, original_json = cli.ot.emd2, cli._returned_solver_note, cli.cpu.json_bytes
    primaries = []

    def altered(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        if kwargs.get("log"):
            distance, log = result
            return distance, (malformed_log if malformed_log is not None else
                              dict(log, result_code=status))
        return result

    def capture_primary(error, *args):
        primaries.append(error)
        original_note(error, *args)

    def unavailable(value):
        if formatting_failure and type(value) is dict and "request_id" in value:
            raise OSError("synthetic status-export formatter unavailable")
        return original_json(value)

    altered.__signature__ = inspect.signature(original_solve)
    monkeypatch.setattr(cli.ot, "emd2", altered)
    monkeypatch.setattr(cli, "_returned_solver_note", capture_primary)
    monkeypatch.setattr(cli.cpu, "json_bytes", unavailable)
    error = None
    try:
        (cli.explicit_pairs if phase == "pair" else cli.complete_rows)(tiny.state)
    except BaseException as caught:
        error = caught
    finally:
        monkeypatch.setattr(cli.cpu, "json_bytes", original_json)
    cells = tiny.state.pair_probes if phase == "pair" else tiny.state.row_cells["status-export"]
    return tiny, cells, error, primaries


def write_fil6_cells(tmp_path, monkeypatch, tiny, cells, phase):
    root = tmp_path/"synthetic-cells"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    header = cli.PAIR_HEADER if phase == "pair" else cli.ROW_HEADER
    name = "pair_probes.csv" if phase == "pair" else "row_cells.csv"
    try:
        bindings = {}
        cli.csv_output(output, name, header, cells, bindings, tiny.state.memory, len(cells))
        raw = (output.out/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == bindings[name]["sha256"]
        parsed = list(cli.producer.csv_rows(
            io.BytesIO(raw), header, "synthetic status cells", compressed=False,
            expanded_limit=64*cli.MIB, row_limit=len(cells), record_limit=8448))
    finally:
        output.close()
    return parsed


@pytest.mark.parametrize("phase", ("pair", "row"))
@pytest.mark.parametrize("status", [
    pytest.param(10**5000, id="positive-huge"),
    pytest.param(-10**5000, id="negative-huge"),
    pytest.param(2**256-1, id="positive-256-bits"),
    pytest.param(-(2**256-1), id="negative-256-bits"),
    pytest.param(2**256, id="positive-257-bits"),
    pytest.param(-2**256, id="negative-257-bits"),
    pytest.param(10**4299, id="positive-4300-digits"),
    pytest.param(-10**4299, id="negative-4300-digits"),
    pytest.param(10**4300, id="positive-4301-digits"),
    pytest.param(-10**4300, id="negative-4301-digits"),
    pytest.param(0, id="zero"), pytest.param(1, id="optimal"), pytest.param(2, id="status-two"),
    pytest.param(-1, id="negative-one"),
    pytest.param(np.int8(-2), id="numpy-int8"),
    pytest.param(np.int16(2), id="numpy-int16"),
    pytest.param(np.int32(-1), id="numpy-int32"),
    pytest.param(np.int64(1), id="numpy-int64"),
    pytest.param(np.longlong(1), id="numpy-longlong"),
    pytest.param(np.uint8(1), id="numpy-uint8"),
    pytest.param(np.uint16(0), id="numpy-uint16"),
    pytest.param(np.uint32(2), id="numpy-uint32"),
    pytest.param(np.uint64(2**64-1), id="numpy-uint64"),
    pytest.param(np.ulonglong(2**64-1), id="numpy-ulonglong"),
])
def test_fil6_c4_status_exports_through_fixed_csv_and_diagnostic(tmp_path, monkeypatch, phase, status):
    limit = sys.get_int_max_str_digits()
    tiny, cells, error, primaries = fil6_c4_native_cell(monkeypatch, phase, status)
    number = int(status)
    exported = number if abs(number).bit_length() <= 256 else None
    parsed = write_fil6_cells(tmp_path, monkeypatch, tiny, cells, phase)
    active = next(c for c in cells if not cli._skip(c))
    serialized = next(c for c in parsed if c["reference_index"] == str(active["reference_index"]))
    assert active["solver_result_code"] == exported
    assert serialized["solver_result_code"] == ("" if exported is None else str(exported))
    if number == 1:
        assert error is None and not primaries and active["computed_valid"]
    else:
        assert type(error) is cli.PilotError and primaries == [error]
        assert active["observation_status"] == "failed" and not active["computed_valid"]
        projected = write_fil6_diagnostic(tmp_path, monkeypatch, error)
        assert all(len(json.dumps(n, ensure_ascii=True).encode("utf8")) <= 4096
                   for n in projected["error"]["notes"])
        if exported is None:
            note = projected["error"]["notes"][0]
            assert "solver_result_code" in note and "not exported" in note
            assert "sign="+("negative" if number < 0 else "positive") in note
            assert "bit_length="+str(abs(number).bit_length()) in note
            assert "256" in note and "null" in note
    assert sys.get_int_max_str_digits() == limit


@pytest.mark.parametrize("phase", ("pair", "row"))
@pytest.mark.parametrize("negative", (False, True))
def test_fil6_c4_omission_survives_formatter_failure(tmp_path, monkeypatch, phase, negative):
    status = -10**5000 if negative else 10**5000
    tiny, cells, error, primaries = fil6_c4_native_cell(
        monkeypatch, phase, status, formatting_failure=True)
    assert type(error) is cli.PilotError and primaries == [error]
    parsed = write_fil6_cells(tmp_path, monkeypatch, tiny, cells, phase)
    assert parsed[-1]["solver_result_code"] == ""
    note = write_fil6_diagnostic(tmp_path, monkeypatch, error)["error"]["notes"][0]
    assert "formatting unavailable" in note and "status-export" in note
    assert "builtins.int" in note and "not exported" in note
    assert "bit_length=16610" in note and "sign="+("negative" if negative else "positive") in note
    assert "null" in note and len(json.dumps(note, ensure_ascii=True).encode("utf8")) <= 4096


@pytest.mark.parametrize("phase", ("pair", "row"))
@pytest.mark.parametrize("string_key", (False, True))
@pytest.mark.parametrize("formatting_failure", (False, True))
def test_fil6_c4_hostile_log_key_preserves_primary_and_serializers(
        tmp_path, monkeypatch, phase, string_key, formatting_failure):
    class Key:
        comparisons = 0

        def __hash__(self):
            return hash("result_code")

        def __eq__(self, other):
            self.comparisons += 1
            raise RuntimeError("synthetic hostile-key equality")

    class StringKey(str):
        comparisons = 0
        __hash__ = str.__hash__
        __eq__ = Key.__eq__

    key = StringKey("result_code") if string_key else Key()
    tiny, cells, error, primaries = fil6_c4_native_cell(
        monkeypatch, phase, 1, malformed_log={key: None}, formatting_failure=formatting_failure)
    assert type(error) is cli.PilotError and str(error) == "OT log: exact keys required"
    assert primaries == [error] and key.comparisons == 0
    parsed = write_fil6_cells(tmp_path, monkeypatch, tiny, cells, phase)
    assert parsed[-1]["solver_result_code"] == "" and not cells[-1]["computed_valid"]
    note = write_fil6_diagnostic(tmp_path, monkeypatch, error)["error"]["notes"][0]
    assert "status-export" in note and "builtins.dict" in note and "field unavailable" in note
    assert len(json.dumps(note, ensure_ascii=True).encode("utf8")) <= 4096
    if formatting_failure:
        assert "formatting unavailable" in note


def test_fil6_c4_malformed_log_and_request_do_not_scan_unsupported_containers():
    class Uninspectable(dict):
        def items(self):
            raise AssertionError("unsupported container must not be scanned")

        def get(self, name, default=None):
            raise AssertionError("unsupported container lookup")

    primary = cli.PilotError("synthetic validation")
    cli._returned_solver_note(primary, Uninspectable(request()), 1,
                              Uninspectable(result_code=10**5000))
    note = primary.__notes__[0]
    assert "unavailable" in note and "Uninspectable" in note
    assert len(json.dumps(note, ensure_ascii=True).encode("utf8")) <= 4096


def test_empty_selection_records_no_prepared_array(monkeypatch):
    tiny = Tiny(monkeypatch)
    mask = tiny.state.bundle.masks[0]
    mask.backing = Backing(np.zeros(6, bool))
    mask.counts = tuple(SimpleNamespace(retained_tiles=0) for _ in range(3))
    event = cli.prepare(tiny.state, tiny.raws[0], request())
    assert event.rows == 0 and event.array is None
    assert tiny.state.preparations[0]["prepared_dtype"] is None


def test_both_budget_links_reuse_actual_arrays(monkeypatch):
    pairs = [request("b5"), request("b100", budget=100)]
    tiny = Tiny(monkeypatch, spec(pairs))
    observations = []
    for req in pairs:
        q = tiny.state.query_event(req)
        r = cli.prepare(tiny.state, tiny.raws[1], req)
        obs = tiny.state.observation(req, 1)
        cli.compute_pair(tiny.state, req, q, r, obs, scalar_control=True)
        observations.append((req, q, r, obs))
        assert tiny.state.query_event(req) is q
    assert observations[0][1].rows == 2 and observations[1][1].rows == 1
    with pytest.raises(cli.PilotError, match="association"):
        cli.compute_pair(tiny.state, observations[1][0], observations[0][1], observations[1][2],
                         tiny.state.observation(observations[1][0], 1), scalar_control=False)
    req, q, r, obs = observations[0]
    repeated = cli.prepare(tiny.state, tiny.raws[1], req)
    assert repeated.preparation_id != r.preparation_id
    with pytest.raises(cli.PilotError):
        cli.validate_preparation(tiny.state, replace(repeated, preparation_id=r.preparation_id), req, 1)
    assert obs["query_preparation_id"] == q.preparation_id
    assert obs["reference_preparation_id"] == r.preparation_id


def test_full_tie_descriptors_and_short_rows(monkeypatch):
    tiny = Tiny(monkeypatch, n=15)
    ordered = [{"reference_index": i, "distance_hex": 1.0.hex()} for i in range(15)]
    relevance = [i % 2 == 0 for i in range(15)]
    for cutoff in (5, 10):
        result = cli.boundary(tiny.state, ordered, relevance, cutoff)
        assert result["equal_count"] == 15 and result["strict_count"] == 0
        assert len(result["selected_ids"]) == cutoff
    short = cli.boundary(tiny.state, ordered[:2], relevance[:2], 5)
    assert short["status"] == "no_kth_boundary" and short["distance_hex"] is None
    assert cli.kernel.query_metrics([True, False], 5, "returned").conditional_precision == 0.2


def test_selected_metadata_declaration_exactness(monkeypatch):
    tiny = Tiny(monkeypatch)
    bundle = tiny.state.bundle
    bundle.masks = bundle.masks[:1]
    bundle.feature_records = tuple(
        SimpleNamespace(slide_index=i, rows=2,
            consumer_asset=SimpleNamespace(asset_id="feature:"+s.slide_id) if i < 2 else None)
        for i, s in enumerate(bundle.slides))
    derived = cli.validate_bundle(tiny.plan, bundle)
    assert derived["feature_indices"] == [0, 1] and derived["mask_views"] == [[0, 5]]
    bundle.feature_records[2].consumer_asset = SimpleNamespace(asset_id="feature:102")
    with pytest.raises(cli.PilotError, match="extra feature"):
        cli.validate_bundle(tiny.plan, bundle)


def test_output_original_bytes_and_semantics(tmp_path, monkeypatch):
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    bindings, memory = {}, cli.WorkingMemory(64*cli.MIB)
    try:
        records = [{"value": (-0.0).hex()}, {"value": np.nextafter(1.0, 2.0).item().hex()}]
        cli.csv_output(output, "values.csv", ("value",), records, bindings, memory, 2)
        cli.json_output(output, "value.json", {"value": 0.2}, bindings, memory)
        output.check(bindings)
        assert len(bindings) == 2
    finally:
        output.close()


def test_original_same_size_mutation_rejected(tmp_path, monkeypatch):
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    original = output.readback

    def changed(name, limit, semantic, **kw):
        path = output.out/name
        data = path.read_bytes().replace(b'"value":1', b'"value":2')
        with path.open("r+b") as handle:
            handle.write(data)
        return original(name, limit, semantic, **kw)

    monkeypatch.setattr(output, "readback", changed)
    try:
        with pytest.raises(Exception, match="original serialized"):
            cli.json_output(output, "value.json", {"value": 1}, {}, cli.WorkingMemory(64*cli.MIB))
    finally:
        output.close()


@pytest.mark.parametrize("fault", (OSError, KeyboardInterrupt))
def test_release_then_error_removes_owned_complete(tmp_path, monkeypatch, fault):
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    cli.json_output(output, "COMPLETE.json", {"synthetic": True}, {}, cli.WorkingMemory(64*cli.MIB))
    original = os.close
    error = fault("released")

    def close(fd):
        original(fd)
        raise error

    with monkeypatch.context() as change:
        change.setattr(cli.producer.os, "close", close)
        with pytest.raises(fault) as caught:
            output.close()
    assert caught.value is error
    try:
        output.fail(error, [])
        assert not (output.out/"COMPLETE.json").exists()
        assert json.loads((output.out/"FAILED").read_text())["fit_warnings"] == []
    finally:
        output.close(error)


def test_reader_bound_cli_dependency_gate():
    """No mock-only normal CLI success can satisfy the Main integration gate."""
    expected = os.environ.get("R21_FILTER_STABLE_READER_SHA")
    if not expected:
        pytest.skip("Lead stable Main reader version not released to this consumer")
    reader = cli._reader()
    assert hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest() == expected


_INTEGRATION_HANDOFF = {
    "prosper/evaluation/r2_1_hoptimus_mosaic_reader.py":
        (47074, "599e1c2643ce87e2d98daac2acccfd41a82297454f6ec5b82a7416ca50b5af24"),
    "tests/test_r2_1_hoptimus_mosaic_reader.py":
        (55684, "55996ff594af6e1768bf03b957896e2c919a34181d4cb599c595ac67ddd53f31"),
    "examples/r2_1_hoptimus_filter_retrieval.json":
        (4296, "36cf182b4a07c887e7070c0ee75ab0d6a85c67566b9af44ecab42bbfc665b42b"),
}


def integration_source_handshake():
    assert os.environ.get("R21_FILTER_STABLE_READER_SHA") == _INTEGRATION_HANDOFF[
        "prosper/evaluation/r2_1_hoptimus_mosaic_reader.py"][1]
    package = Path(__file__).resolve().parents[1]
    for name, (size, digest) in _INTEGRATION_HANDOFF.items():
        raw = (package/name).read_bytes()
        assert len(raw) == size and hashlib.sha256(raw).hexdigest() == digest
    return {role: {"asset_id": "code:"+role, "role": "executing_code",
                   "path": str(path.resolve()), **cli.binding(path.read_bytes())}
            for role, path in cli.code_paths().items()}


def integration_publish(path, data, identity=None, role=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    result = {"path": str(path), **cli.binding(data)}
    if identity is not None:
        result.update(asset_id=identity, role=role)
    return result


@pytest.fixture(scope="module")
def actual_reader_integration(tmp_path_factory):
    source_bindings = integration_source_handshake()
    root = tmp_path_factory.mktemp("synthetic_actual_reader")
    inputs = root/"inputs"
    inputs.mkdir()
    main_test = Path(__file__).with_name("test_r2_1_hoptimus_mosaic_reader.py")
    loaded = importlib.util.spec_from_file_location("_filter29_stable_main_fixture", main_test)
    module = importlib.util.module_from_spec(loaded)
    sys.modules[loaded.name] = module
    loaded.loader.exec_module(module)
    packet = module.Packet(inputs/"mosaic")
    for row in packet.population_rows:
        row[2] = row[0]
    packet.sync_population()
    raws = []
    for index, count in enumerate(packet.rows):
        raw = np.zeros((count, 1536), np.float32, order="F" if index == 1 else "C")
        for row in range(count):
            raw[row, index*4+row] = 1
        raws.append(raw)
        encoded = pickle.dumps((raw, packet.coordinates[index]), protocol=4)
        packet.feature_bytes[index] = encoded
        pin = cli.binding(encoded)
        packet.spec["features"][index].update(pin)
        entry = next(b for b in packet.inputs["bindings"] if b["asset_id"] == "feature:s"+str(index))
        entry.update(pin)
        packet.offset_rows[index][-1] = pin["sha256"]
    packet.files["slide_offsets.csv"] = module.csv_bytes(module.reader.OFFSET_HEADER, packet.offset_rows)
    details = packet.summary["details"]
    sizes = [len(value) for value in packet.feature_bytes.values()]
    details.update(consumed_feature_bytes=sum(sizes), minimum_feature_bytes_with_final_sweep=2*sum(sizes),
                   maximum_encoded_feature_bytes=max(sizes))
    for fold, budget in module.reader.VIEW_ORDER:
        name = f"fold{fold}_budget{budget}"
        mask = np.load(io.BytesIO(packet.files[name+"_mask.npy"]), allow_pickle=False)
        if fold == 1:
            mask[-1] = False
        packet.files[name+"_mask.npy"] = module.npy(mask)
        labels = np.where(mask, 1, 2).astype("<i8")
        probabilities = np.zeros((len(mask), 5), "<f4")
        probabilities[np.arange(len(mask)), labels-1] = 1
        packet.files[name+"_predictions.npy"] = module.npy(labels)
        packet.files[name+"_probabilities.npy"] = module.npy(probabilities)
        for count in packet.count_rows:
            if (count[0], count[1]) == (fold, budget):
                offset = packet.offset_rows[count[2]]
                retained = int(mask[offset[5]:offset[6]].sum())
                count[8], count[9] = retained, count[7]-retained
                count[13], count[14] = retained, count[7]-retained
                count[-1] = "nonempty" if retained else "zero_predicted_tumor"
    packet.sync_counts()
    zeros = sum(count[-1] == "zero_predicted_tumor" for count in packet.count_rows)
    packet.summary["counts"].update(zero_mask_cells=zeros, nonzero_mask_cells=30-zeros)
    packet.publish()
    for name, data in packet.files.items():
        if not (packet.root/name).exists():
            integration_publish(packet.root/name, data)
    pairs, rows = [], []
    for method in cli.METHODS:
        for query in (0, 1):
            for selection, budget in (("literal_all_tissue", None), ("all_true", None),
                                      ("filtered", 5), ("filtered", 100)):
                pairs.append(request(f"p-{method}-{query}-{selection}-{budget}",
                                     method, selection, budget, query, 2))
            for selection, budget in (("literal_all_tissue", None), ("filtered", 5), ("filtered", 100)):
                rows.append(row_request(f"r-{method}-{query}-{selection}-{budget}",
                                        method, selection, budget, query))
        for selection, budget in (("literal_all_tissue", None), ("all_true", None),
                                  ("filtered", 5), ("filtered", 100)):
            pairs.append(request(f"excluded-{method}-{selection}-{budget}",
                                 method, selection, budget, 0, 1))
    plan = spec(pairs, rows)
    controls = {}
    for field in cli.MOSAIC_CONTROLS:
        controls[field] = asdict(getattr(packet.bundle_assets, field))
    controls["masks"] = [asdict(value) for value in packet.bundle_assets.masks]
    controls["features"] = [asdict(value) for value in packet.bundle_assets.features]
    plan.update(run_id="synthetic_actual_reader_only", code_files=source_bindings,
                mosaic_bundle_assets=controls, mosaic_expected=asdict(packet.expected))
    metadata = cli.cpu.csv_bytes(("Tumor.group", "Slide.Scan.File", "Sample.ID", "UID.Subject"), [
        {"Tumor.group": p[5], "Slide.Scan.File": p[3], "Sample.ID": "synthetic"+str(p[0]),
         "UID.Subject": p[4]} for p in packet.population_rows])
    baseline = inputs/"baseline"
    sources = {"metadata": {"path": "/synthetic/historical/metadata.csv", **cli.binding(metadata)}}
    matrices = {}
    generator = cli.transport.OTSlideEmbeddingGenerator(
        metric="cosine", reg=0, feature_dim=1536, n_workers=1, verbose=False, max_iter=1000000)
    for method in cli.METHODS:
        prepared = ([generator.process_single_slide(raw, "s"+str(i))["features"]
                     for i, raw in enumerate(raws)] if method == "ot_cosine" else
                    [cli.bob.features_to_barcodes(raw) for raw in raws])
        values = np.empty((3, 3), np.float64)
        for i in range(3):
            for j in range(3):
                values[i, j] = (generator.compute_ot_distance(prepared[i], prepared[j])
                                if method == "ot_cosine" else
                                cli.bob.hamming_min_median(prepared[i], prepared[j]))
        data = pickle.dumps(pd.DataFrame(values, index=["s0", "s1", "s2"],
                                        columns=["s0", "s1", "s2"]), protocol=4)
        matrices[method] = integration_publish(baseline/(method+".pkl"), data,
                                             "baseline:matrix:"+method, "baseline_matrix")
        sources["matrix_"+method] = {"path": "/synthetic/historical/"+method+".pkl", **cli.binding(data)}
    producer_spec = cli.cpu.json_bytes({"schema_version": 1, "task": "hoptimus_main",
        "run_id": "synthetic_baseline", "expected": {"raw_slides": 3}, "sources": sources})
    original_bindings = [{"kind": "spec", "name": "spec", "path": "/synthetic/historical/spec.json",
                          **cli.binding(producer_spec)}]
    original_bindings += [{"kind": "source", "name": name, **value} for name, value in sources.items()]
    baseline_inputs = cli.cpu.json_bytes({"schema_version": 1, "task": "hoptimus_main",
        "run_id": "synthetic_baseline", "spec_sha256": cli.binding(producer_spec)["sha256"],
        "bindings": original_bindings})
    output_registry = {name: cli.binding(b"") for name in cli.MAIN_OUTPUTS}
    output_registry["input_bindings.json"] = cli.binding(baseline_inputs)
    complete = cli.cpu.json_bytes({"schema_version": 1, "task": "hoptimus_main",
        "scope": "hoptimus_main_saved_rank_evaluation_only", "run_id": "synthetic_baseline",
        "spec_sha256": cli.binding(producer_spec)["sha256"], "outputs": output_registry})
    historical_key = "/synthetic/historical/COMPLETE.json"
    review = cli.cpu.json_bytes({"schema_version": 1, "disposition": "accepted",
        "scope": "Actual main saved-source evaluation and filter/perturbation preparation bundles only",
        "branches": {"main": {"disposition": "accepted"}},
        "artifacts": {historical_key: cli.binding(complete)["sha256"]},
        "synthetic_fixture_only": True})
    payloads = {"review": review, "complete": complete, "spec": producer_spec,
                "input_bindings": baseline_inputs, "metadata": metadata}
    plan["baseline_assets"] = {name: integration_publish(
        baseline/(name+".json" if name != "metadata" else "metadata.csv"), data,
        "baseline:"+("review57" if name == "review" else name), "baseline_"+name)
        for name, data in payloads.items()}
    plan["baseline_assets"].update(matrices=matrices, historical_review_complete_key=historical_key)
    plan["baseline_expected"] = {name+"_sha256": cli.binding(data)["sha256"]
                                 for name, data in payloads.items() if name != "metadata"}
    plan["baseline_expected"]["run_id"] = "synthetic_baseline"
    base = cli.fit._runtime_profile()
    logical = [sys.executable, cli.pd.__file__, cli.threadpoolctl.__file__, cli.ot.__file__,
               inspect.getfile(cli.ot.dist), cli.simplex.__file__, cli.emd_wrap.__file__,
               cli.np._core._multiarray_umath.__file__]
    logical += [entry["path"] for entry in base["implementation"].values()]
    logical += [pool["filepath"] for pool in base["native_pools"]]
    implementations = [{"asset_id": f"runtime:impl:synthetic{index:03d}",
                        "role": "runtime_implementation", "path": path,
                        **cli.binding(Path(path).read_bytes())}
                       for index, path in enumerate(sorted({str(Path(p).resolve()) for p in logical}))]
    profile = cli.runtime_snapshot(implementations)["profile"]
    expected_profile = integration_publish(inputs/"runtime_profile.json", cli.cpu.json_bytes(profile),
                                           "runtime:expected", "runtime_profile")
    plan["runtime_assets"] = {"expected_profile": expected_profile, "implementations": implementations}
    specification = integration_publish(inputs/"pilot_spec.json", cli.cpu.json_bytes(plan))
    guard_dir = inputs/"caller_tripwire"
    guard_dir.mkdir()
    guard_source = (
        "import importlib.abc, os, sys\n"
        "TOKEN='r2_1_hoptimus_input_observation_cli'\n"
        "class Guard(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path, target=None):\n"
        "  if TOKEN in fullname: raise AssertionError('changing caller import forbidden')\n"
        "def audit(event,args):\n"
        " if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and TOKEN in os.fsdecode(args[0]):\n"
        "  raise AssertionError('changing caller source/cache read forbidden')\n"
        "sys.meta_path.insert(0,Guard())\n"
        "sys.addaudithook(audit)\n")
    integration_publish(guard_dir/"sitecustomize.py", guard_source.encode("ascii"))
    integration_publish(root/"provenance.json", cli.cpu.json_bytes({
        "scope": "trusted_small_synthetic_receipts_no_actual105_or57_data_no_fitting",
        "counts": asdict(packet.counts), "source_bindings": source_bindings,
        "raw_layouts": [cli._layout(raw) for raw in raws],
        "fixture_runtime_only_not_software_resource_acceptance": True}))
    fixture = SimpleNamespace(root=root, inputs=inputs, packet=packet, plan=plan,
                              spec=specification, source_bindings=source_bindings,
                              guard_dir=guard_dir, raws=raws)
    yield fixture
    assert integration_source_handshake() == source_bindings


def integration_arguments(fixture, output):
    return ["--spec", fixture.spec["path"], "--spec-sha256", fixture.spec["sha256"],
            "--out", str(output), "--out-root", os.environ["PROSPER_OUT"]]


def integration_check_complete(output):
    assert not (output/"FAILED").exists() and not (output/"DIAGNOSTIC.json").exists()
    assert {p.name for p in output.iterdir()} == {*cli.ARTIFACTS, "COMPLETE.json"}
    marker = json.loads((output/"COMPLETE.json").read_bytes())
    assert marker["disposition"] == "success" and set(marker["outputs"]) == set(cli.ARTIFACTS)
    for name, pin in marker["outputs"].items():
        data = (output/name).read_bytes()
        assert cli.binding(data) == pin
        if name.endswith(".json"):
            cli.cpu.strict_json(data)
        else:
            list(csv.DictReader(io.StringIO(data.decode("utf8"))))
    return marker


def test_actual_reader_pristine_normal_cli(actual_reader_integration):
    fixture = actual_reader_integration
    output = fixture.root/"pristine_output"
    command = [sys.executable, "-m", "prosper.evaluation.r2_1_hoptimus_filter_retrieval_cli",
               *integration_arguments(fixture, output)]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(fixture.guard_dir)+os.pathsep+str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(command, capture_output=True, env=environment, check=False)
    integration_publish(fixture.root/"pristine_stdout.log", completed.stdout)
    integration_publish(fixture.root/"pristine_stderr.log", completed.stderr)
    integration_publish(fixture.root/"pristine_command.json",
                        cli.cpu.json_bytes({"argv": command, "returncode": completed.returncode}))
    assert completed.returncode == 0, completed.stderr.decode("utf8")
    marker = integration_check_complete(output)
    assert json.loads(completed.stdout) == marker
    summary = json.loads((output/"summary.json").read_bytes())
    assert summary["promotion"]["eligible"]
    assert summary["completed_counts"]["pair_requests"] == len(fixture.plan["pair_requests"])
    assert summary["completed_counts"]["reference_cells"] == 3*len(fixture.plan["row_requests"])
    assert any(row["status"] == "empty_query" for row in summary["row_results"])
    assert any(row["metrics"][0]["short_list"] for row in summary["row_results"] if row["status"] == "returned")


@pytest.mark.parametrize("collection", ("consumed_controls", "consumed_arrays"))
@pytest.mark.parametrize("checks", ((), ("size",), ("shape", "coordinates"),
                                  ("shape", "coordinates", "shape")))
def test_consumption_json_projection_actual_main_records(tmp_path, monkeypatch, collection, checks):
    reader = cli._reader()
    source = reader.ConsumptionRecord("synthetic:record", "synthetic_record", "/synthetic/only",
                                      0, hashlib.sha256(b"").hexdigest(), checks)
    bundle = SimpleNamespace(
        expected=SimpleNamespace(review_sha256="1"*64, complete_sha256="2"*64, spec_sha256="3"*64),
        feature_records=(), consumed_controls=(), consumed_arrays=(), referenced_only=())
    setattr(bundle, collection, (source,))
    state = SimpleNamespace(bundle=bundle, decode_counts={}, final=[], baseline=None,
                            started="synthetic_start", spec={"run_id": "synthetic_record"})
    record = cli.inputs_record(state, "4"*64, {})
    projected = record["mosaic_provenance"][collection][0]
    expected = {**asdict(source), "semantic_checks": list(checks)}
    assert projected == expected and type(projected["semantic_checks"]) is list
    assert set(projected) == {"asset_id", "logical_name", "consumer_path", "producer_size",
                              "producer_sha256", "semantic_checks"}
    assert cli.cpu.json_bytes(projected) == cli.cpu.json_bytes(asdict(source))
    assert cli.cpu.strict_json(cli.cpu.json_bytes(record)) == record
    root = tmp_path/"synthetic_projection"
    root.mkdir()
    monkeypatch.setenv("PROSPER_OUT", str(root))
    output = cli.producer._Output(root/"attempt", root, [], [])
    output.claim()
    try:
        bindings = {}
        cli.json_output(output, "input_bindings.json", record, bindings, cli.WorkingMemory(64*cli.MIB))
        assert (output.out/"input_bindings.json").read_bytes() == cli.cpu.json_bytes(record)
    finally:
        output.close()
    projected["semantic_checks"].append("observer_only")
    assert source.semantic_checks == checks and type(source.semantic_checks) is tuple
    with pytest.raises(FrozenInstanceError):
        source.semantic_checks = ()


def test_actual_reader_consumption_roundtrip(actual_reader_integration):
    """Updated live witness; old failing source and artifacts remain preserved."""
    fixture = actual_reader_integration
    with fixture.packet.guard as guard:
        bundle = cli._reader().load_mosaic_bundle(
            guard, bundle_assets=fixture.packet.bundle_assets, expected=fixture.packet.expected)
        final = guard.final_bindings()
    state = SimpleNamespace(bundle=bundle, decode_counts={}, final=final, baseline=None,
                            started="synthetic_start", spec=fixture.plan)
    record = cli.inputs_record(state, fixture.spec["sha256"], cli.code_paths())
    decoded = cli.cpu.strict_json(cli.cpu.json_bytes(record))
    assert decoded == record
    for collection in ("consumed_controls", "consumed_arrays"):
        originals = getattr(bundle, collection)
        assert originals
        for original, projected in zip(originals, record["mosaic_provenance"][collection], strict=True):
            assert type(original.semantic_checks) is tuple
            assert projected["semantic_checks"] == list(original.semantic_checks)
            assert cli.cpu.json_bytes(projected) == cli.cpu.json_bytes(asdict(original))
    output = cli.producer._Output(fixture.root/"consumption_roundtrip",
                                 os.environ["PROSPER_OUT"], [], [])
    output.claim()
    try:
        bindings = {}
        cli.json_output(output, "input_bindings.json", record, bindings, cli.WorkingMemory(64*cli.MIB))
        assert (output.out/"input_bindings.json").read_bytes() == cli.cpu.json_bytes(record)
    finally:
        output.close()
    integration_publish(fixture.root/"consumption_roundtrip.json", cli.cpu.json_bytes({
        "scope": "actual_Main_tuple_API_JSON_native_Filter_projection",
        "controls": len(bundle.consumed_controls), "arrays": len(bundle.consumed_arrays),
        "same_serialized_record_bytes": True, "immutable_Main_records_preserved": True,
        "unchanged_strict_writer_readback_passed": True}))


def integration_read_cells(output):
    return {phase: list(csv.DictReader(io.StringIO((output/name).read_text())))
            for phase, name in (("pair", "pair_probes.csv"), ("row", "row_cells.csv"))}


def integration_array(root, name, array):
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=(1, 0), allow_pickle=False)
    return {**integration_publish(root/(name+".npy"), buffer.getvalue()),
            "dtype": array.dtype.str, "shape": list(array.shape),
            "strides_at_computation": list(array.strides),
            "process_local_array_identity": id(array)}


def integration_event_oracle(fixture, output, trace, cells):
    preparations = {row["preparation_id"]: row for row in csv.DictReader(
        io.StringIO((output/"preparations.csv").read_text()))}
    requests = {row["request_id"]: row for row in
                fixture.plan["pair_requests"]+fixture.plan["row_requests"]}
    observed = {}
    for frame in trace:
        request_record = requests[frame["request_id"]]
        assert frame["method"] == request_record["method"]
        assert frame["selection"] == request_record["selection"]
        assert frame["budget_percent"] == request_record["budget_percent"]
        assert frame["query_fold"] == request_record["query_fold"]
        for side, index in (("query", request_record["query_index"]),
                            ("reference", frame["reference_index"])):
            event = preparations[frame[side+"_preparation_id"]]
            assert int(event["slide_index"]) == index, side+" preparation slide"
            for name in ("method", "selection", "budget_percent", "query_fold"):
                expected = request_record[name]
                assert event[name] == ("" if expected is None else str(expected)), side+" preparation view"
            raw = fixture.raws[index]
            if request_record["selection"] == "filtered":
                view = f"fold{request_record['query_fold']}_budget{request_record['budget_percent']}_mask.npy"
                mask = np.load(fixture.packet.root/view, allow_pickle=False)
                offset = fixture.packet.offset_rows[index]
                raw = raw[mask[offset[5]:offset[6]]]
            elif request_record["selection"] == "all_true":
                raw = raw[np.ones(len(raw), bool)]
            expected_array = (cli.transport.OTSlideEmbeddingGenerator(
                metric="cosine", reg=0, feature_dim=1536, n_workers=1, verbose=False,
                max_iter=1000000).process_single_slide(raw, "oracle")["features"]
                if request_record["method"] == "ot_cosine" else cli.bob.features_to_barcodes(raw))
            description = frame[side+"_array"]
            data = Path(description["path"]).read_bytes()
            assert cli.binding(data) == {k: description[k] for k in ("size", "sha256")}
            actual = np.load(io.BytesIO(data), allow_pickle=False)
            assert actual.dtype.str == description["dtype"] == expected_array.dtype.str
            assert list(actual.shape) == description["shape"] == list(expected_array.shape)
            assert actual.tobytes(order="C") == expected_array.tobytes(order="C"), side+" actual array/view"
        key = (frame["phase"], frame["request_id"], frame["reference_index"])
        assert key not in observed
        observed[key] = frame
    checked = {"pair": 0, "row": 0}
    for phase, records in cells.items():
        for cell in records:
            key = (phase, cell["request_id"], int(cell["reference_index"]))
            if cell["computed_valid"] == "1":
                frame = observed.pop(key)
                assert cell["query_preparation_id"] == frame["query_preparation_id"], "query event link"
                assert cell["reference_preparation_id"] == frame["reference_preparation_id"], "reference event link"
                checked[phase] += 1
            else:
                assert key not in observed
    assert not observed and checked["pair"] and checked["row"]
    return checked


def test_actual_reader_instrumented_arrays_events_and_corruption_oracles(
        actual_reader_integration, monkeypatch, capsys):
    fixture = actual_reader_integration
    output, trace_root = fixture.root/"instrumented_output", fixture.root/"observational_trace"
    trace_root.mkdir()
    trace, raw_calls = [], []
    reader = cli._reader()
    original_compute, original_raw = cli.compute_pair, reader.load_raw_slide

    def observe_raw(guard, bundle, index):
        raw = original_raw(guard, bundle, index)
        raw_calls.append({"slide_index": index, "layout": asdict(raw.layout),
                          "raw_array_identity": id(raw.raw), "coordinates": raw.coordinates.tolist()})
        return raw

    def observe_compute(state, req, query, reference, cell, *, scalar_control):
        assert state.prepared_events[query.preparation_id] is query
        assert state.prepared_events[reference.preparation_id] is reference
        assert state.query_prepared[(req["query_index"], cli.recipe(req))] is query
        assert state.stage in ("pair_probe", "row_pair")
        ordinal = len(trace)
        frame = {"phase": "row" if state.stage == "row_pair" else "pair",
            "request_id": req["request_id"], "reference_index": cell["reference_index"],
            "method": req["method"], "selection": req["selection"],
            "budget_percent": req["budget_percent"], "query_fold": req["query_fold"],
            "query_index": req["query_index"], "query_preparation_id": query.preparation_id,
            "reference_preparation_id": reference.preparation_id,
            "query_array": integration_array(trace_root, f"call{ordinal:03d}_query", query.array),
            "reference_array": integration_array(trace_root, f"call{ordinal:03d}_reference", reference.array)}
        trace.append(frame)
        result = original_compute(state, req, query, reference, cell, scalar_control=scalar_control)
        assert cell["query_preparation_id"] == query.preparation_id
        assert cell["reference_preparation_id"] == reference.preparation_id
        return result

    with monkeypatch.context() as patch:
        patch.setattr(reader, "load_raw_slide", observe_raw)
        patch.setattr(cli, "compute_pair", observe_compute)
        assert cli.main(integration_arguments(fixture, output)) == 0
    marker = integration_check_complete(output)
    assert json.loads(capsys.readouterr().out) == marker
    cells = integration_read_cells(output)
    checked = integration_event_oracle(fixture, output, trace, cells)
    groups = {}
    for frame in trace:
        key = (frame["query_index"], frame["method"], frame["selection"],
               frame["budget_percent"], frame["query_fold"])
        groups.setdefault(key, set()).add(frame["query_preparation_id"])
    assert all(len(ids) == 1 for ids in groups.values())
    assert all(sum(call["slide_index"] == index for call in raw_calls) == 1 for index in (0, 1))
    assert any(call["layout"]["f_contiguous"] and not call["layout"]["c_contiguous"] for call in raw_calls)
    assert any(call["layout"]["c_contiguous"] and not call["layout"]["f_contiguous"] for call in raw_calls)
    assert {f["budget_percent"] for f in trace if f["selection"] == "filtered"} == {5, 100}
    assert {f["method"] for f in trace} == set(cli.METHODS)
    assert {f["selection"] for f in trace} == {"literal_all_tissue", "all_true", "filtered"}
    assert np.load(fixture.packet.root/"fold0_budget5_mask.npy", allow_pickle=False)[-1]
    assert not np.load(fixture.packet.root/"fold1_budget5_mask.npy", allow_pickle=False)[-1]
    integration_publish(trace_root/"trace.json", cli.cpu.json_bytes(trace))
    integration_publish(trace_root/"raw_decoder_observations.json", cli.cpu.json_bytes(raw_calls))

    row_frame = next(f for f in trace if f["phase"] == "row" and f["query_index"] == 0
                     and f["method"] == "yottixel" and f["budget_percent"] == 100)
    budget5 = next(f for f in trace if f["query_index"] == 0 and f["method"] == "yottixel"
                   and f["budget_percent"] == 5)
    wrong_trace, wrong_cells = copy.deepcopy(trace), copy.deepcopy(cells)
    changed = next(f for f in wrong_trace if f == row_frame)
    changed["query_preparation_id"], changed["query_array"] = (
        budget5["query_preparation_id"], budget5["query_array"])
    changed_cell = next(c for c in wrong_cells["row"] if c["request_id"] == row_frame["request_id"]
                        and int(c["reference_index"]) == row_frame["reference_index"])
    changed_cell["query_preparation_id"] = budget5["query_preparation_id"]
    integration_publish(trace_root/"wrong_budget_trace.json", cli.cpu.json_bytes(wrong_trace))
    integration_publish(trace_root/"wrong_budget_cells.json", cli.cpu.json_bytes(wrong_cells))
    with pytest.raises(AssertionError, match="query preparation view"):
        integration_event_oracle(fixture, output, wrong_trace, wrong_cells)

    repeated = next(f for f in trace if f["phase"] == "pair" and f["query_index"] == 0
                    and f["method"] == row_frame["method"] and f["budget_percent"] == 100
                    and f["selection"] == "filtered")
    assert repeated["reference_preparation_id"] != row_frame["reference_preparation_id"]
    wrong_cells = copy.deepcopy(cells)
    changed_cell = next(c for c in wrong_cells["row"] if c["request_id"] == row_frame["request_id"]
                        and int(c["reference_index"]) == row_frame["reference_index"])
    changed_cell["reference_preparation_id"] = repeated["reference_preparation_id"]
    integration_publish(trace_root/"wrong_repeated_event_cells.json", cli.cpu.json_bytes(wrong_cells))
    with pytest.raises(AssertionError, match="reference event link"):
        integration_event_oracle(fixture, output, trace, wrong_cells)
    integration_publish(trace_root/"oracle_results.json", cli.cpu.json_bytes({
        "scope": "separate_observational_run_not_pristine_accounting",
        "checked": checked, "both_row_event_links_verified": True,
        "actual_arrays_verified_for_both_budgets_methods_and_routes": True,
        "cached_query_events_reused_without_extra_preparation": True,
        "wrong_budget_cache_copy_rejected": True, "wrong_repeated_event_link_copy_rejected": True,
        "query_fold_reference_mask_distinguished_from_reference_own_fold": True}))


def integration_copy_inputs(fixture, name):
    root = fixture.root/name
    shutil.copytree(fixture.inputs, root)

    def alias(value):
        if type(value) is dict:
            return {k: alias(v) for k, v in value.items()}
        if type(value) is list:
            return [alias(v) for v in value]
        if type(value) is str and value.startswith(str(fixture.inputs)+os.sep):
            return str(root)+value[len(str(fixture.inputs)):]
        return value

    plan = alias(fixture.plan)
    path = root/"pilot_spec.json"
    path.write_bytes(cli.cpu.json_bytes(plan))
    return SimpleNamespace(spec={"path": str(path), **cli.binding(path.read_bytes())},
                           plan=plan, inputs=root)


def integration_check_failure(output, error):
    assert not (output/"COMPLETE.json").exists() and not (output/"COMPLETE").exists()
    failed = json.loads((output/"FAILED").read_bytes())
    diagnostic = json.loads((output/"DIAGNOSTIC.json").read_bytes())
    assert failed["fit_warnings"] == []
    assert failed["error_type"] == diagnostic["error"]["type"] == type(error).__name__
    assert failed["message"] == diagnostic["error"]["message"] == str(error)
    assert not diagnostic["custody"]["output_custody_verified"]
    return diagnostic


def test_actual_reader_copied_input_final_sweep_failure(actual_reader_integration, monkeypatch):
    fixture = actual_reader_integration
    copied = integration_copy_inputs(fixture, "copied_input_negative")
    output = fixture.root/"copied_input_failure_output"
    target = Path(copied.plan["mosaic_bundle_assets"]["features"][-1]["asset"]["path"])
    before = target.read_bytes()
    original = cli.assets.AssetGuard.final_bindings
    primaries, mutated = [], []

    def late_mutation(guard):
        if not mutated:
            target.write_bytes(before+b"explicit synthetic late mutation")
            mutated.append(True)
        try:
            return original(guard)
        except BaseException as error:
            primaries.append(error)
            raise

    with monkeypatch.context() as patch:
        patch.setattr(cli.assets.AssetGuard, "final_bindings", late_mutation)
        with pytest.raises(cli.assets.AssetError) as caught:
            cli.main(integration_arguments(copied, output))
    assert primaries == [caught.value] and mutated
    diagnostic = integration_check_failure(output, caught.value)
    assert not diagnostic["custody"]["final_sweep_completed"]
    original_target = Path(fixture.plan["mosaic_bundle_assets"]["features"][-1]["asset"]["path"])
    assert original_target.read_bytes() == before
    integration_publish(fixture.root/"copied_input_failure_oracle.json", cli.cpu.json_bytes({
        "primary_identity_preserved": True, "mutation_confined_to_copied_feature": str(target),
        "original_input_unchanged": True, "no_success_marker": True,
        "diagnostic": diagnostic}))


def test_actual_reader_copied_output_late_release_and_secondary_failure(
        actual_reader_integration, monkeypatch):
    fixture = actual_reader_integration
    copied = integration_copy_inputs(fixture, "copied_output_negative_inputs")
    output = fixture.root/"copied_output_failure"
    original_close, original_fail = cli.producer._Output.close, cli.producer._Output.fail
    primaries, secondaries = [], []

    def late_close(owned, primary=None):
        original_close(owned, primary)
        if owned.out == output and primary is None and not primaries:
            assert (output/"COMPLETE.json").exists()
            try:
                (output/"deliberately_missing_late_parent"/"leaf").read_bytes()
            except FileNotFoundError as error:
                primaries.append(error)
                raise

    def failed_marker_then_secondary(owned, error, fit_warnings):
        original_fail(owned, error, fit_warnings)
        if owned.out == output:
            try:
                (output/"deliberately_missing_secondary_parent"/"leaf").read_bytes()
            except FileNotFoundError as additional:
                secondaries.append(additional)
                raise

    with monkeypatch.context() as patch:
        patch.setattr(cli.producer._Output, "close", late_close)
        patch.setattr(cli.producer._Output, "fail", failed_marker_then_secondary)
        with pytest.raises(FileNotFoundError) as caught:
            cli.main(integration_arguments(copied, output))
    assert primaries == [caught.value] and len(secondaries) == 1
    diagnostic = integration_check_failure(output, caught.value)
    assert diagnostic["stage"] == "directory_release"
    assert diagnostic["custody"]["final_sweep_completed"]
    assert len(diagnostic["custody"]["secondary_errors"]) == 1
    assert any("missing_secondary_parent" in note for note in diagnostic["error"]["notes"])
    integration_publish(fixture.root/"copied_output_failure_oracle.json", cli.cpu.json_bytes({
        "scope": "explicit_synthetic_filesystem_ENOENT_after_owned_release_and_FAILED",
        "primary_identity_preserved": True, "secondary_failure_retained": True,
        "owned_COMPLETE_written_then_removed": True, "no_consumable_success": True,
        "diagnostic": diagnostic}))
