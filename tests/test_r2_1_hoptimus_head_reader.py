"""Synthetic accepted-head graphs only; no producer fit or real model reads."""

from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import pickletools
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from prosper.evaluation import r2_1_hoptimus_asset_io as assets
from prosper.evaluation import r2_1_hoptimus_fit as fit
from prosper.evaluation import r2_1_hoptimus_head_reader as reader


ROLES = ("acceptance", "registry", "pilot_acceptance", "complete", "spec",
         "input_bindings", "summary", "diagnostics", "model", "head_plan")
ARTIFACTS = ("head.pkl", "fit_diagnostics.json", "input_bindings.json", "summary.json")
OUTPUTS = (*ARTIFACTS, "held_out_rows.csv.gz", "probabilities.npy", "predictions.npy", "counts.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def binding(data):
    return dict(size=len(data), sha256=sha(data))


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("reader/fixture must not fit")
    monkeypatch.setattr(fit, "fit_head", forbidden)
    monkeypatch.setattr(fit.SGDClassifier, "fit", forbidden)


def make_graph(tmp_path, fold=0, budget=5, fit_warning=False):
    key = f"fold{fold}_budget{budget}"
    historical = f"/historical/{'pilots' if (fold, budget) in ((0, 5), (1, 100)) else 'remaining'}/{key}"
    ids = reader.HeadBundleIds(
        **{role: role for role in ROLES},
        registry_complete_key=historical + "/COMPLETE.json",
        producer_spec_path=f"/historical/declarations/{key}_spec.json",
        registry_artifact_keys=tuple((name, historical + "/" + name) for name in ARTIFACTS),
    )
    plans = []
    for f in range(5):
        for b in (5, 100):
            chosen = 2 if b == 5 else 40
            plans.append(dict(
                held_out_fold=f, budget_percent=b, pool_count=200, selected_count=5 * chosen,
                selected_fraction=5 * chosen / 200,
                class_stats=[dict(tissue_label=label, pool_count=40, selected_count=chosen,
                                  fraction=chosen / 40, balanced_weight=1.0)
                             for label in range(1, 6)],
                training_tile_ids_sha256=sha(f"order {f} {b}".encode()),
            ))
    plan = dict(schema_version=1, run_id="synthetic_CPU_filter_family",
                task="hoptimus_filter_manifest", classes=[1, 2, 3, 4, 5], heads=plans)
    selected = next(head for head in plans if head["held_out_fold"] == fold and head["budget_percent"] == budget)
    estimator = fit.SGDClassifier(**fit._parameters({label: 1.0 for label in range(1, 6)}))
    estimator.classes_ = np.arange(1, 6, dtype=np.int64)
    estimator.coef_ = np.zeros((5, 1536), dtype=np.float32)
    estimator.intercept_ = np.zeros(5, dtype=np.float32)
    estimator.n_features_in_, estimator.n_iter_, estimator.t_ = 1536, 1, 20.0
    runtime = fit._runtime_profile()
    recorded_warnings = ([dict(category="sklearn.exceptions.ConvergenceWarning",
                               message="synthetic retained fit warning", filename="synthetic.py", lineno=7)]
                         if fit_warning else [])
    status = "iteration_limit_warning_retained" if fit_warning else "no_warning_not_optimality_claim"
    diagnostic = dict(
        schema_version=1, held_out_fold=fold, budget_percent=budget,
        selected_tiles=selected["selected_count"], selected_slides=1, selected_patients=1,
        pool_count=200, selected_fraction=selected["selected_fraction"],
        class_support=[dict(tissue_label=row["tissue_label"], pool_tiles=row["pool_count"],
                            selected_tiles=row["selected_count"], selected_slides=1,
                            selected_patients=1, selected_fraction=row["fraction"],
                            balanced_weight=row["balanced_weight"]) for row in selected["class_stats"]],
        ordered_tile_ids_sha256=selected["training_tile_ids_sha256"], feature_sha256=sha(b"synthetic features"),
        feature_shape=[selected["selected_count"], 1536], feature_dtype="<f4",
        parameters=estimator.get_params(deep=False), classes=[1, 2, 3, 4, 5], finite_state=True,
        coefficient_shape=[5, 1536], coefficient_dtype="<f4", intercept_shape=[5],
        intercept_dtype="<f4", n_features_in=1536, n_iter=1, t=20.0,
        n_iter_semantics="maximum_across_five_one_vs_rest_fits", warnings=recorded_warnings,
        iteration_limit_warning=bool(recorded_warnings), convergence_status=status, runtime=runtime,
        fitted_state_sha256=fit._fingerprint(estimator),
        limits=["selected_manifest_cannot_prove_unseen_pool_membership",
                "bare_matrix_cannot_prove_row_identity_or_file_provenance",
                "state_fingerprint_is_not_a_model_serializer_or_history_guard"],
    )
    run = f"synthetic_annotation_{key}"
    expected = dict(annotation_tiles=400, annotation_slides=2, annotation_patients=2,
                    union_patients=6, feature_files=2, training_tiles=selected["selected_count"],
                    training_slides=1, training_patients=1, held_out_tiles=200,
                    held_out_slides=1, held_out_patients=1)
    descriptor = dict(path="/historical/not-reread", size=1, sha256=sha(b"x"))
    spec = dict(schema_version=1, run_id=run, task="hoptimus_annotation_head",
                parameters=dict(held_out_fold=fold, budget_percent=budget, prediction_batch_rows=4096),
                expected=expected, features=[dict(slide_id="one"), dict(slide_id="two")],
                code_files={"fit": dict(descriptor)}, sources={"head_plan": dict(descriptor)})
    inputs = dict(schema_version=1, run_id=run, task=spec["task"], spec_sha256="a" * 64,
                  head=dict(held_out_fold=fold, budget_percent=budget), bindings=[],
                  consumed_cpu_artifacts={})
    summary = dict(
        schema_version=1, run_id=run, task=spec["task"], counts=dict(expected),
        policy=dict(held_out_fold=fold, budget_percent=budget, classes=[1, 2, 3, 4, 5],
                    recipe="r2_1_fixed_sgd_ovr_v1", feature_dtype="float32", prediction_batch_rows=4096,
                    decision="probability_argmax_class1", scope="annotation_patient_held_out_head_only",
                    tuning="none", encoder_replay="not_performed_existing_annotation_bank"),
        details=dict(fitted_state_sha256=diagnostic["fitted_state_sha256"],
                     training_tile_ids_sha256=selected["training_tile_ids_sha256"],
                     serialization_protocol=4, held_out_prediction_rows=200,
                     fit_warnings=recorded_warnings, convergence_status=status),
        limitations=["synthetic-only"],
    )
    complete = dict(schema_version=1, run_id=run, task=spec["task"],
                    scope="annotation_patient_held_out_head_only", spec_sha256="a" * 64, outputs={})
    registry = dict(
        schema_version=1,
        scope="Lead eight-output plus retained-pilot control/row/count reconciliation; not production aggregation",
        exact_ten_tuple_and_once_per_budget_coverage=True,
        heads={f"fold{f}_budget{b}": dict(
            complete_sha256=sha(f"other {f} {b}".encode()),
            retained_pilot=(f, b) in ((0, 5), (1, 100))) for f in range(5) for b in (5, 100)},
        output_hashes={},
    )
    pilot = dict(
        schema_version=1, disposition="accepted", review_kind="actual_outputs_and_observed_resources",
        scope="Actual source/model/prediction/count/resource outcomes of the two prescribed pilot heads only",
        complete_bindings={},
    )
    acceptance = dict(
        schema_version=1, disposition="accepted",
        review_kind="actual_outputs_observed_resources_and_combined_ten_head_coverage",
        scope="Eight new prescribed annotation heads plus exact combined coverage with two unchanged accepted pilots",
        lead_reconciliation_sha256="a" * 64, retained_pilot_actual_acceptance_sha256="a" * 64,
        complete_and_artifact_bindings=dict(
            file="lead_actual_output_reconciliation.json", sha256="a" * 64,
            fields=["heads.*.complete_sha256", "output_hashes"], complete_records=10),
    )
    graph = SimpleNamespace(
        root=tmp_path, fold=fold, budget=budget, key=key, ids=ids, estimator=estimator,
        diagnostic=diagnostic, plan=plan, spec=spec, inputs=inputs, summary=summary,
        complete=complete, registry=registry, pilot=pilot, acceptance=acceptance, paths={}, data={},
    )
    save_graph(graph)
    return graph


def save_graph(graph, *, model=None, spec_entry=True, plan_links=True, outputs=True,
               registry=True, pilot=True, acceptance=True, model_diagnostics=None):
    data = {}
    data["head_plan"] = encode(graph.plan)
    if plan_links:
        graph.spec["sources"]["head_plan"] = dict(path="/historical/CPU/head_plan.json",
                                                 **binding(data["head_plan"]))
    data["spec"] = encode(graph.spec)
    if spec_entry:
        graph.inputs["spec_sha256"] = graph.complete["spec_sha256"] = sha(data["spec"])
        graph.inputs["bindings"] = [
            dict(asset_id="spec", role="spec", path=graph.ids.producer_spec_path,
                 resolved_path=graph.ids.producer_spec_path, **binding(data["spec"]), device=999, inode=888),
            dict(asset_id="source:head_plan", role="head_plan", path="/historical/CPU/head_plan.json",
                 resolved_path="/historical/CPU/head_plan.json", **binding(data["head_plan"]), device=777, inode=666),
        ]
    if plan_links:
        graph.inputs["consumed_cpu_artifacts"]["head_plan.json"] = binding(data["head_plan"])
        for entry in graph.inputs["bindings"]:
            if entry["asset_id"] == "source:head_plan":
                entry.update(binding(data["head_plan"]))
    data["input_bindings"] = encode(graph.inputs)
    data["summary"], data["diagnostics"] = encode(graph.summary), encode(graph.diagnostic)
    if model is None:
        head = fit.FittedHead(graph.estimator, graph.diagnostic if model_diagnostics is None else model_diagnostics)
        data["model"] = pickle.dumps(head, protocol=4)
    else:
        data["model"] = model
    if outputs:
        graph.complete["outputs"] = {name: dict(size=1, sha256=sha(b"x")) for name in OUTPUTS}
        for name, role in zip(ARTIFACTS, ("model", "diagnostics", "input_bindings", "summary")):
            graph.complete["outputs"][name] = binding(data[role])
    data["complete"] = encode(graph.complete)
    if registry:
        graph.registry["heads"][graph.key]["complete_sha256"] = sha(data["complete"])
        graph.registry["output_hashes"] = {graph.ids.registry_complete_key: sha(data["complete"])}
        for name, key in graph.ids.registry_artifact_keys:
            graph.registry["output_hashes"][key] = graph.complete["outputs"][name]["sha256"]
    if pilot:
        graph.pilot["complete_bindings"] = {
            key + "_attempt001/COMPLETE.json": graph.registry["heads"][key]["complete_sha256"]
            for key in ("fold0_budget5", "fold1_budget100")}
    data["registry"], data["pilot_acceptance"] = encode(graph.registry), encode(graph.pilot)
    if acceptance:
        graph.acceptance["lead_reconciliation_sha256"] = sha(data["registry"])
        graph.acceptance["complete_and_artifact_bindings"]["sha256"] = sha(data["registry"])
        graph.acceptance["retained_pilot_actual_acceptance_sha256"] = sha(data["pilot_acceptance"])
    data["acceptance"] = encode(graph.acceptance)
    graph.root.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        path = graph.root / (role + ".bin")
        path.write_bytes(data[role])
        graph.paths[role] = path
    graph.data = data
    return graph


def make_guard(graph, *, limits=None, declarations=None):
    limits = {} if limits is None else limits
    policies = {role: assets.AssetPolicy("bytes", limits.get(role, (4 if role == "head_plan" else 2) * 1024**2))
                for role in ROLES}
    entries = [assets.AssetSpec(getattr(graph.ids, role), role, graph.paths[role],
                               sha(graph.data[role]), len(graph.data[role])) for role in ROLES]
    if declarations is not None:
        entries = declarations(entries)
    return assets.AssetGuard(entries, base_dir=graph.root, policies=policies)


def load(graph, guard, ids=None):
    return reader.load_saved_annotation_head(
        guard, bundle_ids=graph.ids if ids is None else ids,
        held_out_fold=graph.fold, budget_percent=graph.budget)


@pytest.mark.parametrize("fold,budget", [(f, b) for f in range(5) for b in (5, 100)])
def test_valid_ten_routes_caller_ownership_and_prediction(tmp_path, fold, budget):
    graph = make_graph(tmp_path, fold, budget)
    with make_guard(graph) as guard:
        head, provenance = load(graph, guard)
        assert type(head) is fit.FittedHead and guard.state == "active"
        assert provenance.held_out_fold == fold and provenance.budget_percent == budget
        assert provenance.run_id == graph.spec["run_id"]
        assert provenance.producer_spec_path == graph.ids.producer_spec_path
        assert provenance.artifact_sha256 == tuple((role, sha(graph.data[role])) for role in ROLES)
        assert provenance.registry_artifact_keys == graph.ids.registry_artifact_keys
        assert json.loads(provenance.diagnostics_json) == json.loads(encode(graph.diagnostic))
        assert json.loads(provenance.consumer_runtime_json)["distribution_versions"] == \
            graph.diagnostic["runtime"]["distribution_versions"]
        predictions = fit.predict_head(head, np.zeros((2, 1536), dtype=np.float32))
        assert predictions.labels.tolist() == [1, 1]
        assert predictions.probabilities.shape == (2, 5)
        assert len(guard.final_bindings()) == 10 and guard.state == "sealed"
    assert guard.state == "closed"


def test_deep_immutable_public_provenance_and_bundle_pair_order(tmp_path):
    graph = make_graph(tmp_path)
    ids = replace(graph.ids, registry_artifact_keys=tuple(reversed(graph.ids.registry_artifact_keys)))
    assert ids.registry_artifact_keys == graph.ids.registry_artifact_keys
    with make_guard(graph) as guard:
        _, provenance = load(graph, guard, ids)
        with pytest.raises(FrozenInstanceError):
            provenance.run_id = "other"
        with pytest.raises(TypeError):
            provenance.artifact_sha256[0] = ("acceptance", "a" * 64)
        detached = asdict(provenance)
        detached["run_id"] = "other"
        assert provenance.run_id == graph.spec["run_id"]
        assert all(type(value) in (str, int, tuple) for value in detached.values())


@pytest.mark.parametrize("field,value", [
    ("model", "diagnostics"), ("model", ""), ("model", True),
    ("registry_complete_key", "relative/COMPLETE.json"), ("producer_spec_path", "relative"),
    ("registry_artifact_keys", []), ("registry_artifact_keys", (("head.pkl", "/x"),)),
])
def test_invalid_bundle_ids(tmp_path, field, value):
    graph = make_graph(tmp_path)
    with pytest.raises(reader.HeadReaderError):
        replace(graph.ids, **{field: value})


@pytest.mark.parametrize("fold,budget", [(True, 5), (0.0, 5), (-1, 5), (5, 5), (0, True), (0, 5.0), (0, 10)])
def test_requested_tuple_strict_before_any_read(tmp_path, monkeypatch, fold, budget):
    graph = make_graph(tmp_path)
    guard = make_guard(graph)
    monkeypatch.setattr(guard, "read_verified", lambda *a: pytest.fail("invalid tuple read"))
    with pytest.raises(reader.HeadReaderError):
        reader.load_saved_annotation_head(guard, bundle_ids=graph.ids,
                                         held_out_fold=fold, budget_percent=budget)


@pytest.mark.parametrize("role", [role for role in ROLES if role != "model"])
@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_all_control_schema_versions_are_strict(tmp_path, role, version):
    graph = make_graph(tmp_path)
    names = {"input_bindings": "inputs", "diagnostics": "diagnostic", "head_plan": "plan",
             "pilot_acceptance": "pilot"}
    getattr(graph, names.get(role, role))["schema_version"] = version
    save_graph(graph)
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError, match="schema_version"):
        load(graph, guard)


@pytest.mark.parametrize("role", ROLES)
def test_reader_limits_are_pre_read_policy_limits(tmp_path, monkeypatch, role):
    graph = make_graph(tmp_path)
    cap = (4 if role == "head_plan" else 2) * 1024**2

    def large_declaration(entries):
        return [assets.AssetSpec(e.asset_id, e.role, e.path, e.sha256, cap + 1)
                if e.role == role else e for e in entries]

    guard = make_guard(graph, declarations=large_declaration)
    opened = []
    original = assets.os.open

    def checked(path, flags):
        opened.append(Path(path))
        assert Path(path) != graph.paths[role], "oversized role reached file open"
        return original(path, flags)

    monkeypatch.setattr(assets.os, "open", checked)
    with guard, pytest.raises(assets.AssetError, match="encoded size"):
        load(graph, guard)
    assert graph.paths[role] not in opened


@pytest.mark.parametrize("case", [
    "acceptance-disposition", "acceptance-kind", "acceptance-scope", "registry-review-hash",
    "registry-link-hash", "registry-field-links", "complete-records", "pilot-review-hash",
    "pilot-disposition", "pilot-kind", "pilot-scope", "pilot-complete",
    "registry-scope", "coverage-bool", "missing-tuple", "retained-flag", "complete-head-link",
    "complete-key-link", "artifact-key-link", "artifact-complete-hash", "extra-output", "missing-output",
])
def test_review_registry_and_complete_counterexamples(tmp_path, case):
    graph = make_graph(tmp_path)
    options = {}
    if case.startswith("acceptance-"):
        graph.acceptance[case.removeprefix("acceptance-") if case != "acceptance-kind" else "review_kind"] = "wrong"
    elif case == "registry-review-hash":
        graph.acceptance["lead_reconciliation_sha256"] = "a" * 64
        options["acceptance"] = False
    elif case == "registry-link-hash":
        graph.acceptance["complete_and_artifact_bindings"]["sha256"] = "a" * 64
        options["acceptance"] = False
    elif case == "registry-field-links":
        graph.acceptance["complete_and_artifact_bindings"]["fields"] = ["wrong"]
    elif case == "complete-records":
        graph.acceptance["complete_and_artifact_bindings"]["complete_records"] = 10.0
    elif case == "pilot-review-hash":
        graph.acceptance["retained_pilot_actual_acceptance_sha256"] = "a" * 64
        options["acceptance"] = False
    elif case in ("pilot-disposition", "pilot-kind", "pilot-scope"):
        graph.pilot[case.removeprefix("pilot-") if case != "pilot-kind" else "review_kind"] = "wrong"
    elif case == "pilot-complete":
        graph.pilot["complete_bindings"]["fold0_budget5_attempt001/COMPLETE.json"] = "a" * 64
        options["pilot"] = False
    elif case == "registry-scope":
        graph.registry["scope"] = "wrong"
    elif case == "coverage-bool":
        graph.registry["exact_ten_tuple_and_once_per_budget_coverage"] = 1
    elif case == "missing-tuple":
        del graph.registry["heads"]["fold4_budget100"]
    elif case == "retained-flag":
        graph.registry["heads"]["fold0_budget5"]["retained_pilot"] = 1
    elif case == "complete-head-link":
        graph.registry["heads"][graph.key]["complete_sha256"] = "a" * 64
        options["registry"] = False
    elif case == "complete-key-link":
        graph.registry["output_hashes"][graph.ids.registry_complete_key] = "a" * 64
        options["registry"] = False
    elif case == "artifact-key-link":
        graph.registry["output_hashes"][dict(graph.ids.registry_artifact_keys)["head.pkl"]] = "a" * 64
        options["registry"] = False
    elif case == "artifact-complete-hash":
        graph.complete["outputs"]["head.pkl"]["sha256"] = "a" * 64
        options["outputs"] = False
    elif case == "extra-output":
        graph.complete["outputs"]["extra.json"] = binding(b"extra")
        options["outputs"] = False
    else:
        del graph.complete["outputs"]["counts.json"]
        options["outputs"] = False
    save_graph(graph, **options)
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError):
        load(graph, guard)


@pytest.mark.parametrize("field,value", [("size", True), ("size", 1.0), ("size", -1),
                                        ("sha256", "A" * 64), ("sha256", "x")])
def test_unconsumed_artifact_bindings_still_validated(tmp_path, field, value):
    graph = make_graph(tmp_path)
    graph.complete["outputs"]["counts.json"][field] = value
    save_graph(graph, outputs=False)
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError):
        load(graph, guard)


@pytest.mark.parametrize("case", [
    "run", "task", "scope", "spec-top-digest", "input-top-digest", "spec-entry-digest",
    "spec-entry-size", "spec-entry-path", "duplicate-spec", "alias-spec-role", "duplicate-asset",
    "spec-tuple", "input-tuple", "summary-policy", "summary-count", "plan-input-hash",
    "plan-source-hash", "plan-consumed-hash",
])
def test_producer_identity_and_plan_links(tmp_path, case):
    graph = make_graph(tmp_path)
    options = {}
    if case == "run":
        graph.summary["run_id"] = "other"
    elif case == "task":
        graph.complete["task"] = "hoptimus_main"
    elif case == "scope":
        graph.complete["scope"] = "mosaic_transfer_accepted"
    elif case in ("spec-top-digest", "input-top-digest"):
        target = graph.complete if case == "spec-top-digest" else graph.inputs
        target["spec_sha256"] = "b" * 64
        options["spec_entry"] = False
    elif case.startswith("spec-entry-"):
        field = case.removeprefix("spec-entry-")
        graph.inputs["bindings"][0][{"digest": "sha256"}.get(field, field)] = \
            {"digest": "b" * 64, "size": 1, "path": "/other/spec.json"}[field]
        options["spec_entry"] = False
    elif case == "duplicate-spec":
        graph.inputs["bindings"].append(dict(graph.inputs["bindings"][0]))
        options["spec_entry"] = False
    elif case == "alias-spec-role":
        graph.inputs["bindings"].append({**graph.inputs["bindings"][0], "asset_id": "other"})
        options["spec_entry"] = False
    elif case == "duplicate-asset":
        graph.inputs["bindings"].append(dict(graph.inputs["bindings"][1]))
        options["spec_entry"] = False
    elif case == "spec-tuple":
        graph.spec["parameters"]["held_out_fold"] = True
    elif case == "input-tuple":
        graph.inputs["head"]["budget_percent"] = 5.0
    elif case == "summary-policy":
        graph.summary["policy"]["prediction_batch_rows"] = 4096.0
    elif case == "summary-count":
        graph.summary["counts"]["training_patients"] = True
    elif case == "plan-input-hash":
        graph.inputs["bindings"][1]["sha256"] = "b" * 64
        options.update(spec_entry=False, plan_links=False)
    elif case == "plan-source-hash":
        graph.spec["sources"]["head_plan"]["sha256"] = "b" * 64
        options["plan_links"] = False
    else:
        graph.inputs["consumed_cpu_artifacts"]["head_plan.json"]["sha256"] = "b" * 64
        options["plan_links"] = False
    save_graph(graph, **options)
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError):
        load(graph, guard)


@pytest.mark.parametrize("case", [
    "plan-task", "duplicate-plan", "class-order", "plan-fraction", "plan-weight",
    "diagnostic-fold", "diagnostic-budget", "diagnostic-order", "diagnostic-selected",
    "diagnostic-fraction", "diagnostic-patient", "diagnostic-class-count",
    "diagnostic-class-weight", "diagnostic-state", "diagnostic-shape", "diagnostic-finite",
    "diagnostic-fit-warning-flag",
])
def test_plan_support_order_and_diagnostic_swaps(tmp_path, case):
    graph = make_graph(tmp_path)
    selected = graph.plan["heads"][0]
    if case == "plan-task":
        graph.plan["task"] = "hoptimus_annotation_head"
    elif case == "duplicate-plan":
        graph.plan["heads"][-1] = dict(graph.plan["heads"][0])
    elif case == "class-order":
        selected["class_stats"].reverse()
    elif case == "plan-fraction":
        selected["selected_fraction"] = True
    elif case == "plan-weight":
        selected["class_stats"][0]["balanced_weight"] = 2.0
    elif case == "diagnostic-fold":
        graph.diagnostic["held_out_fold"] = True
    elif case == "diagnostic-budget":
        graph.diagnostic["budget_percent"] = 100
    elif case == "diagnostic-order":
        graph.diagnostic["ordered_tile_ids_sha256"] = "a" * 64
    elif case == "diagnostic-selected":
        graph.diagnostic["selected_tiles"] = 11
    elif case == "diagnostic-fraction":
        graph.diagnostic["selected_fraction"] = "0.05"
    elif case == "diagnostic-patient":
        graph.diagnostic["selected_patients"] = 2
    elif case == "diagnostic-class-count":
        graph.diagnostic["class_support"][0]["selected_tiles"] = 3
    elif case == "diagnostic-class-weight":
        graph.diagnostic["parameters"]["class_weight"][1] = 2.0
    elif case == "diagnostic-state":
        graph.diagnostic["fitted_state_sha256"] = "b" * 64
    elif case == "diagnostic-shape":
        graph.diagnostic["coefficient_shape"] = [True, 1536]
    elif case == "diagnostic-finite":
        graph.diagnostic["finite_state"] = 1
    else:
        graph.diagnostic["iteration_limit_warning"] = 0
    save_graph(graph)
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError):
        load(graph, guard)


def test_CPU_plan_run_is_not_annotation_run(tmp_path):
    graph = make_graph(tmp_path)
    assert graph.plan["run_id"] != graph.spec["run_id"]
    with make_guard(graph) as guard:
        load(graph, guard)


def test_finite_recorded_fit_warning_preserved_not_loader_failure(tmp_path):
    graph = make_graph(tmp_path, fit_warning=True)
    with make_guard(graph) as guard:
        head, provenance = load(graph, guard)
        assert head.diagnostics["warnings"] == graph.diagnostic["warnings"]
        assert json.loads(provenance.diagnostics_json)["iteration_limit_warning"] is True


@pytest.mark.parametrize("categories", [
    ("builtins.RuntimeWarning",),
    ("sklearn.exceptions.ConvergenceWarning", "builtins.RuntimeWarning"),
    ("builtins.RuntimeWarning", "sklearn.exceptions.ConvergenceWarning"),
    ("sklearn.exceptions.InconsistentVersionWarning",),
    ("ConvergenceWarning",),
])
def test_hrdr2_unsupported_or_mixed_recorded_fit_warnings_fail_before_decode(
    tmp_path, monkeypatch, categories,
):
    graph = make_graph(tmp_path, fit_warning=True)
    original = graph.diagnostic["warnings"][0]
    graph.diagnostic["warnings"] = [{**original, "category": category} for category in categories]
    graph.summary["details"]["fit_warnings"] = graph.diagnostic["warnings"]
    save_graph(graph)
    monkeypatch.setattr(reader.legacy, "_pickle",
                        lambda *a: pytest.fail("unsupported recorded fit warning reached decoder"))
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError, match="fit-warning category"):
        load(graph, guard)


def test_hrdr2_multiple_supported_convergence_records_preserved(tmp_path):
    graph = make_graph(tmp_path, fit_warning=True)
    graph.diagnostic["warnings"].append({**graph.diagnostic["warnings"][0],
                                        "message": "second retained convergence record", "lineno": 8})
    graph.summary["details"]["fit_warnings"] = graph.diagnostic["warnings"]
    save_graph(graph)
    with make_guard(graph) as guard:
        head, provenance = load(graph, guard)
        assert head.diagnostics["warnings"] == graph.diagnostic["warnings"]
        assert json.loads(provenance.diagnostics_json)["warnings"] == graph.diagnostic["warnings"]


def test_reader_empty_state_check_calls_no_estimator_prediction(tmp_path, monkeypatch):
    graph = make_graph(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("empty reader validation must not predict")

    monkeypatch.setattr(fit.SGDClassifier, "decision_function", forbidden)
    monkeypatch.setattr(fit.SGDClassifier, "predict_proba", forbidden)
    with make_guard(graph) as guard:
        head, _ = load(graph, guard)
        assert type(head) is fit.FittedHead and guard.state == "active"


def test_returned_mutable_estimator_requires_integrity_path_for_later_batches(tmp_path):
    graph = make_graph(tmp_path)
    with make_guard(graph) as guard:
        head, provenance = load(graph, guard)
        head.estimator.coef_[0, 0] = np.float32(1)
        with pytest.raises(fit.HeadValidationError, match="fingerprint"):
            fit.predict_head(head, np.zeros((1, 1536), dtype=np.float32))
        assert provenance.fitted_state_sha256 == graph.diagnostic["fitted_state_sha256"]


def test_missing_declared_bundle_asset_fails_without_loader(tmp_path, monkeypatch):
    graph = make_graph(tmp_path)
    ids = replace(graph.ids, model="undeclared-model")
    monkeypatch.setattr(reader.legacy, "_pickle", lambda *a: pytest.fail("missing asset decoded"))
    with make_guard(graph) as guard, pytest.raises(assets.AssetError, match="undeclared"):
        load(graph, guard, ids)


def test_runtime_before_loader_and_historical_dispatch_not_equality(tmp_path, monkeypatch):
    graph = make_graph(tmp_path)
    graph.diagnostic["runtime"]["native_pools"] = [{"architecture": "historical-not-current"}]
    save_graph(graph)
    original_runtime, original_load = fit._runtime_profile, reader.legacy._pickle
    calls = []

    def profile():
        calls.append("runtime")
        return original_runtime()

    def decode(*args):
        assert calls == ["runtime"]
        calls.append("decode")
        return original_load(*args)

    monkeypatch.setattr(fit, "_runtime_profile", profile)
    monkeypatch.setattr(reader.legacy, "_pickle", decode)
    with make_guard(graph) as guard:
        _, provenance = load(graph, guard)
        assert json.loads(provenance.consumer_runtime_json)["native_pools"] != \
            graph.diagnostic["runtime"]["native_pools"]
    assert calls == ["runtime", "decode"]


@pytest.mark.parametrize("failure", ["native-pools", "distribution-version", "historical-version"])
def test_runtime_failure_precedes_decoder(tmp_path, monkeypatch, failure):
    graph = make_graph(tmp_path)
    monkeypatch.setattr(reader.legacy, "_pickle", lambda *a: pytest.fail("unsupported runtime decoded model"))
    if failure == "native-pools":
        monkeypatch.setattr(fit, "threadpool_info", lambda: [{"num_threads": 2}])
    elif failure == "distribution-version":
        monkeypatch.setattr(fit.importlib.metadata, "version", lambda name: "0.0.0")
    else:
        graph.diagnostic["runtime"]["distribution_versions"]["numpy"] = "0.0.0"
        save_graph(graph)
    with make_guard(graph) as guard, pytest.raises(ValueError):
        load(graph, guard)


def test_same_size_same_inode_model_change_never_reaches_decoder(tmp_path, monkeypatch):
    graph = make_graph(tmp_path)
    guard = make_guard(graph)
    before = graph.paths["model"].stat()
    changed = bytearray(graph.data["model"])
    changed[-2] ^= 1
    graph.paths["model"].write_bytes(changed)
    after = graph.paths["model"].stat()
    assert (before.st_dev, before.st_ino, before.st_size) == (after.st_dev, after.st_ino, after.st_size)
    monkeypatch.setattr(reader.legacy, "_pickle", lambda *a: pytest.fail("unverified bytes decoded"))
    with guard, pytest.raises(assets.AssetError, match="SHA256"):
        load(graph, guard)


@pytest.mark.parametrize("case", ["protocol", "trailing", "wrapper", "mutable-snapshot", "duplicate-snapshot"])
def test_bad_pickle_framing_wrapper_or_snapshot(tmp_path, case):
    graph = make_graph(tmp_path)
    model = graph.data["model"]
    if case == "protocol":
        model = b"\x80\x03" + model[2:]
    elif case == "trailing":
        model += pickle.dumps(None, protocol=4)
    elif case == "wrapper":
        model = pickle.dumps({"not": "FittedHead"}, protocol=4)
    else:
        head = fit.FittedHead(graph.estimator, graph.diagnostic)
        if case == "mutable-snapshot":
            object.__setattr__(head, "_diagnostic_snapshot", bytearray(head._diagnostic_snapshot))
        else:
            object.__setattr__(head, "_diagnostic_snapshot",
                               b'{"schema_version":1,' + head._diagnostic_snapshot[1:])
        model = pickle.dumps(head, protocol=4)
    save_graph(graph, model=model)
    with make_guard(graph) as guard, pytest.raises(ValueError):
        load(graph, guard)


@pytest.mark.parametrize("case", ["proto4_then_proto5", "proto4_header_with_proto5_opcodes"])
def test_hrdr1_protocol5_counterexamples_rejected_before_unpickle(tmp_path, monkeypatch, case):
    graph = make_graph(tmp_path)
    protocol5 = pickle.dumps(fit.FittedHead(graph.estimator, graph.diagnostic), protocol=5)
    encoded = b"\x80\x04" + (protocol5 if case == "proto4_then_proto5" else protocol5[2:])
    operations = list(pickletools.genops(encoded))
    assert "BYTEARRAY8" in {opcode.name for opcode, _, _ in operations if opcode.proto > 4}
    headers = [argument for opcode, argument, _ in operations if opcode.name == "PROTO"]
    assert headers == ([4, 5] if case == "proto4_then_proto5" else [4])
    save_graph(graph, model=encoded)
    monkeypatch.setattr(reader.legacy, "_pickle",
                        lambda *a: pytest.fail("protocol5 stream reached frozen decoder"))
    monkeypatch.setattr(pickle, "loads", lambda *a, **k: pytest.fail("protocol5 stream unpickled"))
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError, match="protocol"):
        load(graph, guard)


@pytest.mark.parametrize("declaration", [0, 1, 2, 3, 5, 6, 255])
def test_hrdr1_incompatible_later_protocol_declarations_fail_before_decoder(
    tmp_path, monkeypatch, declaration,
):
    graph = make_graph(tmp_path)
    encoded = b"\x80\x04\x80" + bytes((declaration,)) + graph.data["model"][2:]
    save_graph(graph, model=encoded)
    monkeypatch.setattr(reader.legacy, "_pickle",
                        lambda *a: pytest.fail("incompatible declaration reached decoder"))
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError, match="protocol"):
        load(graph, guard)


def test_hrdr1_compatible_repeated_protocol4_and_frozen_trailing_check(tmp_path):
    graph = make_graph(tmp_path)
    save_graph(graph, model=b"\x80\x04" + graph.data["model"])
    with make_guard(graph) as guard:
        head, _ = load(graph, guard)
        assert type(head) is fit.FittedHead
    save_graph(graph, model=graph.data["model"] + pickle.dumps(None, protocol=4))
    with make_guard(graph) as guard, pytest.raises(reader.legacy.AuditError, match="trailing"):
        load(graph, guard)


@pytest.mark.parametrize("encoded", [
    b"\x80\x04", b"\x80\x04\xff", b"\x80\x04\x95\x00",
    b"\x80\x04X\xff\xff\xff\xff", b"\x80\x04\x8c\x01\xff.",
])
def test_hrdr1_malformed_opcode_stream_rejected_without_decoder(tmp_path, monkeypatch, encoded):
    graph = make_graph(tmp_path)
    save_graph(graph, model=encoded)
    monkeypatch.setattr(reader.legacy, "_pickle", lambda *a: pytest.fail("malformed stream decoded"))
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError, match="protocol4"):
        load(graph, guard)


@pytest.mark.parametrize("case", ["FP64", "nonfinite", "reordered", "changed-coef", "changed-recipe", "n_iter"])
def test_empty_prediction_reuses_existing_fitted_state_checks(tmp_path, case):
    graph = make_graph(tmp_path)
    if case == "FP64":
        graph.estimator.coef_ = graph.estimator.coef_.astype(np.float64)
    elif case == "nonfinite":
        graph.estimator.coef_[0, 0] = np.inf
    elif case == "reordered":
        graph.estimator.classes_ = graph.estimator.classes_[::-1].copy()
    elif case == "changed-coef":
        graph.estimator.coef_[0, 0] = 1
    elif case == "changed-recipe":
        graph.estimator.alpha = 0.1
    else:
        graph.estimator.n_iter_ = 1001
    save_graph(graph)
    with make_guard(graph) as guard, pytest.raises(fit.HeadValidationError):
        load(graph, guard)


def test_bound_external_and_internal_diagnostics_must_match(tmp_path):
    graph = make_graph(tmp_path)
    internal = json.loads(encode(graph.diagnostic))
    internal["limits"] = ["different bound diagnostics"]
    save_graph(graph, model_diagnostics=internal)
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError, match="snapshot"):
        load(graph, guard)


@pytest.mark.parametrize("fit_warning", [False, True])
def test_loader_warning_fatal_even_under_outer_ignore(tmp_path, monkeypatch, fit_warning):
    graph = make_graph(tmp_path, fit_warning=fit_warning)
    original = reader.legacy._pickle

    def noisy(*args):
        head = original(*args)
        warnings.warn("synthetic loader warning", RuntimeWarning)
        return head

    monkeypatch.setattr(reader.legacy, "_pickle", noisy)
    with make_guard(graph) as guard, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(reader.HeadReaderError, match="model-loader warnings") as caught:
            load(graph, guard)
        assert "not fit warnings" in "\n".join(caught.value.__notes__)


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_warning_plus_exception_preserves_original_with_notes(tmp_path, monkeypatch, exception_type):
    graph = make_graph(tmp_path)
    original = exception_type("synthetic decoder abort")

    def abort(*args):
        warnings.warn("before loader abort", RuntimeWarning)
        raise original

    monkeypatch.setattr(reader.legacy, "_pickle", abort)
    with make_guard(graph) as guard, pytest.raises(exception_type) as caught:
        load(graph, guard)
    assert caught.value is original
    assert any("before loader abort" in note for note in original.__notes__)
    assert guard.state == "closed"


@pytest.mark.parametrize("outer_filter", ["ignore", "error"])
def test_real_sklearn_version_warning_is_fatal(tmp_path, outer_filter):
    graph = make_graph(tmp_path)
    data = graph.data["model"]
    marker = data.index(b"_sklearn_version")
    start = data.index(b"1.7.0", marker)
    save_graph(graph, model=data[:start] + b"1.6.0" + data[start + 5:])
    with make_guard(graph) as guard, warnings.catch_warnings():
        warnings.simplefilter(outer_filter)
        with pytest.raises(reader.HeadReaderError, match="model-loader warnings") as caught:
            load(graph, guard)
    assert "InconsistentVersionWarning" in "\n".join(caught.value.__notes__)


def test_verified_original_bytes_used_if_source_changes_before_decode(tmp_path, monkeypatch):
    graph = make_graph(tmp_path)
    original_profile = fit._runtime_profile

    def changed_after_reads():
        profile = original_profile()
        changed = bytearray(graph.data["model"])
        changed[-2] ^= 1
        graph.paths["model"].write_bytes(changed)
        return profile

    monkeypatch.setattr(fit, "_runtime_profile", changed_after_reads)
    with make_guard(graph) as guard:
        head, provenance = load(graph, guard)
        assert type(head) is fit.FittedHead
        assert dict(provenance.artifact_sha256)["model"] == sha(graph.data["model"])
        with pytest.raises(assets.AssetError, match="SHA256"):
            guard.final_bindings()


def test_caller_final_sweep_detects_late_source_change(tmp_path):
    graph = make_graph(tmp_path)
    with make_guard(graph) as guard:
        head, _ = load(graph, guard)
        assert guard.state == "active"
        graph.paths["model"].write_bytes(graph.data["model"][:-1] + b"X")
        with pytest.raises(assets.AssetError):
            guard.final_bindings()
        with pytest.raises(assets.AssetError):
            guard.read_verified("model")
        assert type(head) is fit.FittedHead


@pytest.mark.parametrize("role", [role for role in ROLES if role != "model"])
@pytest.mark.parametrize("bad", [
    b'{"schema_version":1,"schema_version":1}', b'{"schema_version":1,"value":NaN}',
    b'{"schema_version":1,"value":1e999}', b"\xff", b"[]",
])
def test_every_json_control_rejects_invalid_finite_structural_input(tmp_path, role, bad):
    graph = make_graph(tmp_path)
    graph.data[role] = bad
    graph.paths[role].write_bytes(bad)
    with make_guard(graph) as guard, pytest.raises(reader.HeadReaderError):
        load(graph, guard)
