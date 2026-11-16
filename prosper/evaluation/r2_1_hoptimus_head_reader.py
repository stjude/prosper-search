"""Reuse trusted, accepted annotation heads; never fit or discover model files.

The caller pins trusted receipts, authenticates the CPU-filter family and current
code/native closure, and owns an active AssetGuard and its final source sweep.
Caller AssetPolicy ceilings MUST precede reads: JSON/model 2 MiB, head plan 4 MiB.
Content links do not authenticate arbitrary reviews or make pickle safe.
Every later scoring batch must use fit.predict_head, not estimator methods.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
import pickletools
import warnings

import numpy as np

from . import feature_bank_audit as legacy
from . import r2_1_hoptimus_asset_io as assets
from . import r2_1_hoptimus_fit as fit
from . import r2_1_hoptimus_io as cpu


_ROLES = (
    "acceptance", "registry", "pilot_acceptance", "complete", "spec",
    "input_bindings", "summary", "diagnostics", "model", "head_plan",
)
_ARTIFACTS = ("head.pkl", "fit_diagnostics.json", "input_bindings.json", "summary.json")
_OUTPUTS = frozenset((*_ARTIFACTS, "held_out_rows.csv.gz", "probabilities.npy",
                      "predictions.npy", "counts.json"))
_TASK = "hoptimus_annotation_head"
_SCOPE = "annotation_patient_held_out_head_only"
_ACCEPTANCE_KIND = "actual_outputs_observed_resources_and_combined_ten_head_coverage"
_ACCEPTANCE_SCOPE = (
    "Eight new prescribed annotation heads plus exact combined coverage with two unchanged accepted pilots"
)
_PILOT_SCOPE = "Actual source/model/prediction/count/resource outcomes of the two prescribed pilot heads only"
_REGISTRY_SCOPE = "Lead eight-output plus retained-pilot control/row/count reconciliation; not production aggregation"
_EXPECTED = frozenset((
    "annotation_tiles", "annotation_slides", "annotation_patients", "union_patients",
    "feature_files", "training_tiles", "training_slides", "training_patients",
    "held_out_tiles", "held_out_slides", "held_out_patients",
))
_MIB = 1024 * 1024
_FIT_WARNING_CATEGORIES = frozenset(("sklearn.exceptions.ConvergenceWarning",))


class HeadReaderError(ValueError):
    """An accepted-head byte link, metadata identity or loading check failed."""


def _require(condition, message):
    if not condition:
        raise HeadReaderError(message)


def _text(value, context):
    _require(type(value) is str and bool(value.strip()) and "\x00" not in value,
             f"{context}: nonblank exact string required")
    return value


def _int(value, context, minimum=0):
    _require(type(value) is int and value >= minimum, f"{context}: exact integer required")
    return value


def _float(value, context):
    _require(type(value) is float and math.isfinite(value), f"{context}: finite float required")
    return value


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value, context):
    _require(type(value) is str and len(value) == 64
             and all(char in "0123456789abcdef" for char in value),
             f"{context}: lowercase SHA256 required")
    return value


def _historical(value, context):
    value = _text(value, context)
    _require(value.startswith("/"), f"{context}: full historical path required")
    return value


def _keys(value, expected, context):
    _require(type(value) is dict and set(value) == set(expected),
             f"{context}: missing/unknown keys")


def _canonical(value):
    try:
        return cpu.json_bytes(value).decode("ascii")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise HeadReaderError(f"invalid finite canonical JSON: {exc}") from exc


def _same(actual, expected, context):
    _require(_canonical(actual) == _canonical(expected), f"{context}: type-sensitive mismatch")


def _json(data, context):
    _require(type(data) is bytes, f"{context}: immutable bytes required")
    try:
        value = cpu.strict_json(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HeadReaderError(f"{context}: invalid strict finite JSON: {exc}") from exc
    _require(type(value) is dict, f"{context}: JSON object required")
    _require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
             f"{context}: integer schema_version 1 required")
    return value


def _tuple(fold, budget):
    _require(_int(fold, "held_out_fold") < 5, "held_out_fold must be 0..4")
    _require(_int(budget, "budget_percent") in (5, 100), "budget_percent must be 5/100")


def _pairs(value, names, context, *, hashes=False):
    _require(type(value) is tuple and len(value) == len(names), f"{context}: immutable tuple required")
    mapped = {}
    for pair in value:
        _require(type(pair) is tuple and len(pair) == 2, f"{context}: immutable pairs required")
        name, item = pair
        _text(name, context)
        _require(name in names and name not in mapped, f"{context}: duplicate/unknown name")
        mapped[name] = _digest(item, context) if hashes else _historical(item, context)
    _require(set(mapped) == set(names), f"{context}: incomplete roles")
    return tuple((name, mapped[name]) for name in names)


@dataclass(frozen=True, slots=True)
class HeadBundleIds:
    acceptance: str
    registry: str
    pilot_acceptance: str
    complete: str
    spec: str
    input_bindings: str
    summary: str
    diagnostics: str
    model: str
    head_plan: str
    registry_complete_key: str
    producer_spec_path: str
    registry_artifact_keys: tuple[tuple[str, str], ...]

    def __post_init__(self):
        ids = [_text(getattr(self, role), role) for role in _ROLES]
        _require(len(set(ids)) == len(ids), "bundle: distinct asset roles require distinct IDs")
        _historical(self.registry_complete_key, "registry_complete_key")
        _historical(self.producer_spec_path, "producer_spec_path")
        pairs = _pairs(self.registry_artifact_keys, _ARTIFACTS, "registry_artifact_keys")
        _require(len({key for _, key in pairs} | {self.registry_complete_key}) == 5,
                 "bundle: ambiguous historical registry keys")
        object.__setattr__(self, "registry_artifact_keys", pairs)


@dataclass(frozen=True, slots=True)
class HeadProvenance:
    held_out_fold: int
    budget_percent: int
    run_id: str
    artifact_sha256: tuple[tuple[str, str], ...]
    registry_complete_key: str
    registry_artifact_keys: tuple[tuple[str, str], ...]
    producer_spec_path: str
    fitted_state_sha256: str
    diagnostics_json: str
    consumer_runtime_json: str

    def __post_init__(self):
        _tuple(self.held_out_fold, self.budget_percent)
        _text(self.run_id, "provenance run_id")
        object.__setattr__(self, "artifact_sha256",
                           _pairs(self.artifact_sha256, _ROLES, "artifact_sha256", hashes=True))
        object.__setattr__(self, "registry_artifact_keys",
                           _pairs(self.registry_artifact_keys, _ARTIFACTS, "registry_artifact_keys"))
        _historical(self.registry_complete_key, "registry_complete_key")
        _historical(self.producer_spec_path, "producer_spec_path")
        _digest(self.fitted_state_sha256, "fitted_state_sha256")
        for name in ("diagnostics_json", "consumer_runtime_json"):
            text = _text(getattr(self, name), name)
            try:
                parsed = cpu.strict_json(text.encode("utf-8"))
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise HeadReaderError(f"{name}: invalid immutable JSON: {exc}") from exc
            _require(type(parsed) is dict and _canonical(parsed) == text,
                     f"{name}: canonical JSON object required")


def _binding(value, context):
    _keys(value, ("size", "sha256"), context)
    _int(value["size"], f"{context} size")
    _digest(value["sha256"], f"{context} digest")


def _matches(value, data, context):
    _binding(value, context)
    _require(value["size"] == len(data) and value["sha256"] == _sha(data),
             f"{context}: bound size/hash mismatch")


def _reviews(controls, data, ids, fold, budget):
    acceptance, registry, pilot = (controls[role] for role in _ROLES[:3])
    _require(acceptance.get("disposition") == "accepted"
             and acceptance.get("review_kind") == _ACCEPTANCE_KIND
             and acceptance.get("scope") == _ACCEPTANCE_SCOPE,
             "combined acceptance: disposition/kind/scope mismatch")
    _require(_digest(acceptance.get("lead_reconciliation_sha256"), "registry review digest")
             == _sha(data["registry"]), "combined acceptance: registry digest mismatch")
    links = acceptance.get("complete_and_artifact_bindings")
    _require(type(links) is dict
             and links.get("file") == "lead_actual_output_reconciliation.json"
             and _digest(links.get("sha256"), "registry link") == _sha(data["registry"]),
             "combined acceptance: artifact registry link mismatch")
    _same(links.get("fields"), ["heads.*.complete_sha256", "output_hashes"], "registry field links")
    _same(links.get("complete_records"), 10, "ten accepted COMPLETE records")
    _require(_digest(acceptance.get("retained_pilot_actual_acceptance_sha256"), "pilot review digest")
             == _sha(data["pilot_acceptance"]), "combined acceptance: pilot digest mismatch")
    _require(pilot.get("disposition") == "accepted"
             and pilot.get("review_kind") == "actual_outputs_and_observed_resources"
             and pilot.get("scope") == _PILOT_SCOPE, "pilot acceptance: disposition/kind/scope mismatch")
    _require(registry.get("scope") == _REGISTRY_SCOPE
             and registry.get("exact_ten_tuple_and_once_per_budget_coverage") is True,
             "registry: wrong scope/coverage")
    heads = registry.get("heads")
    canonical = {f"fold{f}_budget{b}" for f in range(5) for b in (5, 100)}
    _require(type(heads) is dict and set(heads) == canonical, "registry: exact ten head tuples required")
    for name, entry in heads.items():
        _require(type(entry) is dict, f"{name}: registry head object required")
        _digest(entry.get("complete_sha256"), name)
        _require(type(entry.get("retained_pilot")) is bool
                 and entry["retained_pilot"] == (name in ("fold0_budget5", "fold1_budget100")),
                 f"{name}: retained-pilot identity mismatch")
    output_hashes = registry.get("output_hashes")
    _require(type(output_hashes) is dict, "registry: output_hashes object required")
    for key, digest in output_hashes.items():
        _historical(key, "registry output key")
        _digest(digest, "registry output digest")
    selected = f"fold{fold}_budget{budget}"
    digest = _sha(data["complete"])
    _require(heads[selected]["complete_sha256"] == digest
             and output_hashes.get(ids.registry_complete_key) == digest,
             "registry: selected COMPLETE links mismatch")
    pilot_links = pilot.get("complete_bindings")
    _keys(pilot_links, ("fold0_budget5_attempt001/COMPLETE.json",
                       "fold1_budget100_attempt001/COMPLETE.json"), "pilot COMPLETE bindings")
    for name in ("fold0_budget5", "fold1_budget100"):
        _require(_digest(pilot_links[name + "_attempt001/COMPLETE.json"], "pilot COMPLETE digest")
                 == heads[name]["complete_sha256"], "pilot/combined COMPLETE mismatch")
    complete = controls["complete"]
    outputs = complete.get("outputs")
    _keys(outputs, _OUTPUTS, "producer outputs")
    for name, entry in outputs.items():
        _binding(entry, f"producer {name}")
    roles = dict(zip(_ARTIFACTS, ("model", "diagnostics", "input_bindings", "summary")))
    for name, key in ids.registry_artifact_keys:
        role = roles[name]
        _matches(outputs[name], data[role], name)
        _require(output_hashes.get(key) == _sha(data[role]), f"{name}: registry artifact link mismatch")


def _producer(controls, data, ids, fold, budget):
    spec, complete, bindings, summary = (controls[role] for role in
                                         ("spec", "complete", "input_bindings", "summary"))
    run_id = _text(spec.get("run_id"), "producer run_id")
    for name, value in (("spec", spec), ("COMPLETE", complete),
                        ("input_bindings", bindings), ("summary", summary)):
        _require(_text(value.get("run_id"), f"{name} run_id") == run_id
                 and value.get("task") == _TASK, f"{name}: producer run/task mismatch")
    _require(complete.get("scope") == _SCOPE, "COMPLETE: producer scope mismatch")
    digest = _sha(data["spec"])
    _require(_digest(complete.get("spec_sha256"), "COMPLETE spec digest") == digest
             and _digest(bindings.get("spec_sha256"), "bindings spec digest") == digest,
             "producer spec digest headers mismatch")
    _keys(spec, ("schema_version", "run_id", "task", "sources", "code_files",
                 "features", "parameters", "expected"), "producer spec")
    _keys(spec["parameters"], ("held_out_fold", "budget_percent", "prediction_batch_rows"),
          "producer parameters")
    _same(spec["parameters"], dict(held_out_fold=fold, budget_percent=budget,
                                   prediction_batch_rows=4096), "producer tuple")
    _same(bindings.get("head"), dict(held_out_fold=fold, budget_percent=budget), "bindings tuple")
    _keys(spec["expected"], _EXPECTED, "producer expected counts")
    for name, count in spec["expected"].items():
        _int(count, name, 1)
    _require(type(spec["features"]) is list
             and len(spec["features"]) == spec["expected"]["feature_files"],
             "producer feature declaration count mismatch")
    _require(type(spec["code_files"]) is dict, "producer code_files object required")
    entries = bindings.get("bindings")
    _require(type(entries) is list and bool(entries), "producer bindings list required")
    indexed = {}
    for entry in entries:
        _require(type(entry) is dict, "producer binding object required")
        asset_id = _text(entry.get("asset_id"), "producer binding asset_id")
        _text(entry.get("role"), "producer binding role")
        _historical(entry.get("path"), "producer binding lexical path")
        _historical(entry.get("resolved_path"), "producer binding resolved path")
        _int(entry.get("size"), "producer binding size")
        _digest(entry.get("sha256"), "producer binding digest")
        _require(asset_id not in indexed, "duplicate producer asset ID")
        indexed[asset_id] = entry
    specs = [entry for entry in entries if entry["asset_id"] == "spec" or entry["role"] == "spec"]
    _require(len(specs) == 1 and specs[0]["asset_id"] == specs[0]["role"] == "spec",
             "one unambiguous spec/spec binding required")
    spec_entry = specs[0]
    _require(spec_entry["path"] == ids.producer_spec_path
             and spec_entry["size"] == len(data["spec"]) and spec_entry["sha256"] == digest,
             "producer spec entry path/size/hash mismatch")
    source = indexed.get("source:head_plan")
    _require(source is not None and source["role"] == "head_plan", "producer head-plan binding required")
    _matches({key: source[key] for key in ("size", "sha256")}, data["head_plan"],
             "producer head-plan input binding")
    sources = spec["sources"]
    _require(type(sources) is dict and type(sources.get("head_plan")) is dict,
             "producer head-plan declaration required")
    _keys(sources["head_plan"], ("path", "size", "sha256"), "head-plan source declaration")
    _text(sources["head_plan"]["path"], "head-plan source path")
    _matches({key: sources["head_plan"][key] for key in ("size", "sha256")},
             data["head_plan"], "producer head-plan declaration")
    consumed = bindings.get("consumed_cpu_artifacts")
    _require(type(consumed) is dict and "head_plan.json" in consumed, "consumed CPU head plan required")
    _matches(consumed["head_plan.json"], data["head_plan"], "consumed CPU head plan")
    policy = dict(
        held_out_fold=fold, budget_percent=budget, classes=[1, 2, 3, 4, 5],
        recipe="r2_1_fixed_sgd_ovr_v1", feature_dtype="float32",
        prediction_batch_rows=4096, decision="probability_argmax_class1", scope=_SCOPE,
        tuning="none", encoder_replay="not_performed_existing_annotation_bank",
    )
    _same(summary.get("policy"), policy, "producer summary policy")
    _same(summary.get("counts"), spec["expected"], "producer summary counts")
    _require(type(summary.get("details")) is dict, "producer summary details required")
    _same(summary["details"].get("serialization_protocol"), 4, "serialization protocol")
    _same(summary["details"].get("held_out_prediction_rows"), spec["expected"]["held_out_tiles"],
          "held-out prediction count")
    return run_id


def _plan_and_diagnostics(controls, fold, budget):
    plan, diagnostic = controls["head_plan"], controls["diagnostics"]
    _keys(plan, ("schema_version", "run_id", "task", "classes", "heads"), "CPU head plan")
    _text(plan["run_id"], "CPU-filter family run_id")
    _require(plan["task"] == "hoptimus_filter_manifest", "head plan must belong to CPU-filter family")
    _same(plan["classes"], [1, 2, 3, 4, 5], "head plan classes")
    _require(type(plan["heads"]) is list and len(plan["heads"]) == 10, "ten planned heads required")
    planned = {}
    for head in plan["heads"]:
        _keys(head, ("held_out_fold", "budget_percent", "pool_count", "selected_count",
                     "selected_fraction", "class_stats", "training_tile_ids_sha256"), "planned head")
        key = head["held_out_fold"], head["budget_percent"]
        _tuple(*key)
        _require(key not in planned, "duplicate planned tuple")
        planned[key] = head
        pool, selected = (_int(head[name], name, 1) for name in ("pool_count", "selected_count"))
        _require(selected <= pool and
                 _float(head["selected_fraction"], "planned selected_fraction") == selected / pool,
                 "planned count/fraction mismatch")
        _digest(head["training_tile_ids_sha256"], "planned training order")
        rows = head["class_stats"]
        _require(type(rows) is list and len(rows) == 5, "planned five class rows required")
        for label, row in enumerate(rows, 1):
            _keys(row, ("tissue_label", "pool_count", "selected_count", "fraction", "balanced_weight"),
                  "planned class")
            _require(_int(row["tissue_label"], "planned class", 1) == label, "planned class order")
            class_pool = _int(row["pool_count"], "planned class pool", 1)
            class_selected = _int(row["selected_count"], "planned class selected", 1)
            _require(class_selected <= class_pool
                     and _float(row["fraction"], "planned class fraction") == class_selected / class_pool
                     and _float(row["balanced_weight"], "planned class weight") ==
                     selected / (5 * class_selected), "planned class counts/fractions/weights mismatch")
        _require(sum(row["pool_count"] for row in rows) == pool
                 and sum(row["selected_count"] for row in rows) == selected, "planned class totals")
    selected = planned[(fold, budget)]
    _same(diagnostic.get("held_out_fold"), fold, "diagnostic fold")
    _same(diagnostic.get("budget_percent"), budget, "diagnostic budget")
    counts = controls["spec"]["expected"]
    for name, expected in (
        ("selected_tiles", selected["selected_count"]), ("pool_count", selected["pool_count"]),
        ("selected_fraction", selected["selected_fraction"]),
        ("selected_slides", counts["training_slides"]), ("selected_patients", counts["training_patients"]),
    ):
        _same(diagnostic.get(name), expected, f"diagnostic {name}")
    _same(counts["training_tiles"], selected["selected_count"], "training tile count")
    _require(1 <= diagnostic["selected_patients"] <= diagnostic["selected_slides"]
             <= diagnostic["selected_tiles"], "diagnostic head support bounds")
    _same(diagnostic.get("ordered_tile_ids_sha256"), selected["training_tile_ids_sha256"],
          "diagnostic training order")
    details = controls["summary"]["details"]
    _same(details.get("training_tile_ids_sha256"), selected["training_tile_ids_sha256"],
          "summary training order")
    fingerprint = _digest(diagnostic.get("fitted_state_sha256"), "diagnostic state digest")
    _same(details.get("fitted_state_sha256"), fingerprint, "summary state digest")
    for name, expected in (
        ("feature_shape", [selected["selected_count"], 1536]),
        ("feature_dtype", np.dtype(np.float32).str), ("classes", [1, 2, 3, 4, 5]),
        ("coefficient_shape", [5, 1536]), ("coefficient_dtype", np.dtype(np.float32).str),
        ("intercept_shape", [5]), ("intercept_dtype", np.dtype(np.float32).str),
        ("n_features_in", 1536), ("finite_state", True),
    ):
        _same(diagnostic.get(name), expected, f"diagnostic {name}")
    _digest(diagnostic.get("feature_sha256"), "diagnostic feature digest")
    support = diagnostic.get("class_support")
    _require(type(support) is list and len(support) == 5, "diagnostic five class-support rows required")
    weights = {}
    for row, stat in zip(support, selected["class_stats"]):
        _keys(row, ("tissue_label", "pool_tiles", "selected_tiles", "selected_slides",
                    "selected_patients", "selected_fraction", "balanced_weight"), "diagnostic class")
        for name, value in (
            ("tissue_label", stat["tissue_label"]), ("pool_tiles", stat["pool_count"]),
            ("selected_tiles", stat["selected_count"]), ("selected_fraction", stat["fraction"]),
            ("balanced_weight", stat["balanced_weight"]),
        ):
            _same(row[name], value, f"diagnostic class {name}")
        patients = _int(row["selected_patients"], "class selected_patients", 1)
        slides = _int(row["selected_slides"], "class selected_slides", 1)
        _require(patients <= slides <= row["selected_tiles"]
                 and patients <= diagnostic["selected_patients"]
                 and slides <= diagnostic["selected_slides"], "diagnostic class support bounds")
        weights[str(stat["tissue_label"])] = stat["balanced_weight"]
    _require(type(diagnostic.get("parameters")) is dict, "diagnostic parameters required")
    _same(diagnostic["parameters"].get("class_weight"), weights, "diagnostic selected-class weights")
    fit_warnings = diagnostic.get("warnings")
    _require(type(fit_warnings) is list, "fit warning records required")
    for warning in fit_warnings:
        _keys(warning, ("category", "message", "filename", "lineno"), "fit warning")
        category = _text(warning["category"], "fit warning category")
        _require(category in _FIT_WARNING_CATEGORIES,
                 f"unsupported recorded fit-warning category: {category}")
        _require(type(warning["message"]) is str, "fit warning message string required")
        _text(warning["filename"], "fit warning filename")
        _int(warning["lineno"], "fit warning line", 1)
    _same(diagnostic.get("iteration_limit_warning"), bool(fit_warnings), "fit warning flag")
    status = "iteration_limit_warning_retained" if fit_warnings else "no_warning_not_optimality_claim"
    _same(diagnostic.get("convergence_status"), status, "fit convergence status")
    _same(details.get("fit_warnings"), fit_warnings, "summary fit warnings")
    _same(details.get("convergence_status"), status, "summary fit status")
    return fingerprint


def _protocol4(encoded):
    """Non-executing format check; frozen decoding still owns single-object EOF."""
    _require(type(encoded) is bytes and len(encoded) <= 2 * _MIB,
             "head.pkl: immutable bounded protocol4 bytes required")
    _require(encoded[:2] == b"\x80\x04", "head.pkl: protocol4 required")
    try:
        for opcode, argument, position in pickletools.genops(encoded):
            _require(opcode.proto <= 4,
                     f"head.pkl: {opcode.name} at {position} requires protocol{opcode.proto}, not protocol4")
            if opcode.name == "PROTO":
                _require(argument == 4,
                         f"head.pkl: incompatible protocol declaration {argument} at {position}")
    except HeadReaderError:
        raise
    except (ValueError, UnicodeError, OverflowError) as exc:
        raise HeadReaderError(f"head.pkl: invalid protocol4 opcode stream: {exc}") from exc


def load_saved_annotation_head(
    guard: assets.AssetGuard, *, bundle_ids: HeadBundleIds, held_out_fold: int,
    budget_percent: int,
) -> tuple[fit.FittedHead, HeadProvenance]:
    """Load one bound accepted tuple; caller policies enforce pre-read ceilings.

    Validation errors must abort the caller's task. This reader does not poison
    private guard state, finalize custody, certify unconsumed payloads or refit.
    """
    _tuple(held_out_fold, budget_percent)
    _require(type(bundle_ids) is HeadBundleIds, "HeadBundleIds required")
    ids = HeadBundleIds(**{field.name: getattr(bundle_ids, field.name) for field in fields(HeadBundleIds)})
    _require(isinstance(guard, assets.AssetGuard), "active AssetGuard required")
    data = {}
    for role in _ROLES:
        value = guard.read_verified(getattr(ids, role))
        limit = 4 * _MIB if role == "head_plan" else 2 * _MIB
        _require(type(value) is bytes and len(value) <= limit, f"{role}: immutable bounded bytes required")
        data[role] = value
    controls = {role: _json(value, role) for role, value in data.items() if role != "model"}
    _reviews(controls, data, ids, held_out_fold, budget_percent)
    run_id = _producer(controls, data, ids, held_out_fold, budget_percent)
    fingerprint = _plan_and_diagnostics(controls, held_out_fold, budget_percent)
    runtime = fit._runtime_profile()
    historical = controls["diagnostics"].get("runtime")
    _require(type(historical) is dict, "historical fit runtime required")
    _same(historical.get("distribution_versions"), runtime["distribution_versions"],
          "historical/current accepted distribution versions")
    encoded = data["model"]
    _protocol4(encoded)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            head = legacy._pickle(encoded, "accepted saved annotation head")
        except BaseException as exc:
            exc.add_note(f"saved head fold={held_out_fold}, budget={budget_percent}: no retry/refit")
            if caught:
                exc.add_note("Model-loader warnings (not fit warnings): " +
                             _canonical(fit._warning_records(caught)).rstrip())
            raise
    if caught:
        error = HeadReaderError("head.pkl: unexpected model-loader warnings")
        error.add_note("Model-loader warnings (not fit warnings): " +
                       _canonical(fit._warning_records(caught)).rstrip())
        raise error
    _require(type(head) is fit.FittedHead, "head.pkl: exact FittedHead required")
    _require(type(head._diagnostic_snapshot) is bytes, "model diagnostic snapshot must be immutable bytes")
    snapshot = _json(head._diagnostic_snapshot, "model diagnostic snapshot")
    _same(snapshot, controls["diagnostics"], "model immutable diagnostic snapshot")
    _same(head.diagnostics, controls["diagnostics"], "model detached diagnostics")
    fit.predict_head(head, np.empty((0, 1536), dtype=np.float32, order="C"))
    provenance = HeadProvenance(
        held_out_fold=held_out_fold, budget_percent=budget_percent, run_id=run_id,
        artifact_sha256=tuple((role, _sha(data[role])) for role in _ROLES),
        registry_complete_key=ids.registry_complete_key, registry_artifact_keys=ids.registry_artifact_keys,
        producer_spec_path=ids.producer_spec_path, fitted_state_sha256=fingerprint,
        diagnostics_json=_canonical(controls["diagnostics"]), consumer_runtime_json=_canonical(runtime),
    )
    return head, provenance
