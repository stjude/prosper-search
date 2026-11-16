"""R2.1 route (b) A-2 head source: HEAD_REGISTRY.json writer and registry-based head loader.

Replaces only the H-Optimus-1-specific acceptance chain of the frozen shared
head reader (its ``_reviews``: combined/pilot acceptance and the analyst
reconciliation registry). Every other saved-head check is the frozen reader's
own code, imported unchanged: producer run/task/spec/tuple/policy checks
(``_producer``), head-plan/diagnostic consistency (``_plan_and_diagnostics``)
and the protocol-4 framing check (``_protocol4``). Model loading, the runtime
comparison, warning capture and the empty-batch prediction follow the frozen
``load_saved_annotation_head`` tail, with the prediction width taken from the
caller (the authenticated input width) instead of a literal.

Registry assertions (specification 002 section 3 A-2, review 001 M3), per head:
  * registry (held_out_fold, budget_percent) == the head's spec parameters ==
    summary.json policy == input_bindings.json head == a head_plan.json entry;
  * sha256(head.pkl) == the accepted value == the head's COMPLETE.json output
    binding. Accepted value: H-Optimus-1 mode (M-0) the hash bound in the
    reference head's own COMPLETE.json (optionally also the acceptance
    reconciliation registry); UNI2 mode the independently supplied G-5 accepted-heads
    JSON.
  * the head's encoder/bank identity: A-1 outputs carry ``bank``; heads that
    predate A-1 carry none and are admitted only in H-Optimus-1 mode.

Trusted local pickles only; pickle is not safe for untrusted input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import warnings

import numpy as np

from . import feature_bank_audit as legacy
from . import r2_1_hoptimus_fit as fit
from . import r2_1_hoptimus_head_reader as reader
from . import r2_1_hoptimus_io as cpu


TASK = "r2_1_encoder_head_registry"
ACCEPTED_TASK = "r2_1_route_b_g5_accepted_heads"
BANKS = ("h_optimus_1", "uni2")
TUPLES = tuple((fold, budget) for fold in range(5) for budget in (5, 100))
HEAD_ASSETS = ("complete", "spec", "input_bindings", "summary", "diagnostics", "model")
LEAVES = {"complete": "COMPLETE.json", "input_bindings": "input_bindings.json",
          "summary": "summary.json", "diagnostics": "fit_diagnostics.json", "model": "head.pkl"}
CONSUMED_OUTPUTS = {"head.pkl": "model", "fit_diagnostics.json": "diagnostics",
                    "input_bindings.json": "input_bindings", "summary.json": "summary"}
MIB = 1024 * 1024
HEAD_LIMIT = 2 * MIB
PLAN_LIMIT = 4 * MIB
REGISTRY_LIMIT = 2 * MIB
ACCEPTED_SOURCES = {
    "h_optimus_1": "reference_head_complete_json_output_binding",
    "uni2": "g5_accepted_heads_json",
}


class RegistryError(ValueError):
    """A registry declaration, assertion or saved-head check failed."""


def require(condition, message):
    if not condition:
        raise RegistryError(message)


def name_of(key):
    return f"fold{key[0]}_budget{key[1]}"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _binding(data):
    return {"size": len(data), "sha256": _sha(data)}


def _json(data, context):
    try:
        value = cpu.strict_json(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RegistryError(f"{context}: invalid JSON: {exc}") from exc
    require(type(value) is dict, f"{context}: JSON object required")
    return value


def _digest(value, context):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"{context}: lowercase SHA256 required")
    return value


def _tuple(fold, budget, context):
    require(type(fold) is int and type(budget) is int and 0 <= fold < 5 and budget in (5, 100),
            f"{context}: invalid (held_out_fold, budget_percent)")
    return fold, budget


def parse_accepted(data, bank):
    """G-5 accepted-heads JSON -> {fold{f}_budget{b}: head.pkl sha256} (UNI2 mode only)."""
    value = _json(data, "accepted heads")
    cpu.keys(value, {"schema_version", "task", "encoder", "bank", "heads"}, "accepted heads")
    require(value["schema_version"] == 1 and value["task"] == ACCEPTED_TASK,
            "accepted heads: schema/task mismatch")
    require(value["bank"] == bank == value["encoder"], "accepted heads: bank/encoder mismatch")
    heads = value["heads"]
    require(type(heads) is dict and set(heads) == {name_of(key) for key in TUPLES},
            "accepted heads: exactly the ten head tuples required")
    result = {}
    for name, entry in heads.items():
        cpu.keys(entry, {"head.pkl_sha256"}, f"accepted {name}")
        result[name] = _digest(entry["head.pkl_sha256"], f"accepted {name}")
    return result


def _bank_identity(controls, bank):
    """A-1 outputs carry encoder/bank; pre-A-1 reference heads carry neither."""
    present = {role: ("bank" in controls[role], "encoder" in controls[role])
               for role in ("complete", "summary", "input_bindings")}
    if any(any(flags) for flags in present.values()):
        for role in present:
            require(controls[role].get("bank") == bank and controls[role].get("encoder") == bank,
                    f"{role}: head encoder/bank is not {bank}")
        return "declared_by_head"
    require(bank == "h_optimus_1",
            "head declares no encoder/bank; only pre-adapter H-Optimus-1 reference heads may omit it")
    return "pre_adapter_h_optimus_1_reference_head"


def verify_head(data, *, held_out_fold, budget_percent, bank, accepted_sha256, width,
                producer_spec_path):
    """Apply the registry assertions and the frozen reader's head checks; load the model.

    ``data`` maps HEAD_ASSETS roles plus ``head_plan`` to already-authenticated bytes.
    Returns (FittedHead, provenance dict). Any failure aborts the caller; no refit.
    """
    require(bank in BANKS, f"unknown bank {bank!r}")
    fold, budget = _tuple(held_out_fold, budget_percent, "registry")
    require(type(width) is int and width == fit._WIDTH, "head input width must equal fit _WIDTH")
    require(set(data) == {*HEAD_ASSETS, "head_plan"}, "head data roles")
    for role, value in data.items():
        limit = PLAN_LIMIT if role == "head_plan" else HEAD_LIMIT
        require(type(value) is bytes and len(value) <= limit, f"{role}: immutable bounded bytes required")
    controls = {role: _json(value, role) for role, value in data.items() if role != "model"}
    complete, spec, summary, bindings = (controls[r] for r in ("complete", "spec", "summary", "input_bindings"))
    # Registry assertion 1: declared tuple (spec, summary policy, input bindings, COMPLETE spec link, plan).
    parameters = spec.get("parameters")
    require(type(parameters) is dict and parameters.get("held_out_fold") == fold and
            parameters.get("budget_percent") == budget,
            f"registry assertion: {name_of((fold, budget))} != producer spec parameters")
    policy = summary.get("policy")
    require(type(policy) is dict and policy.get("held_out_fold") == fold and
            policy.get("budget_percent") == budget,
            f"registry assertion: {name_of((fold, budget))} != summary.json policy")
    head = bindings.get("head")
    require(type(head) is dict and head.get("held_out_fold") == fold and head.get("budget_percent") == budget,
            f"registry assertion: {name_of((fold, budget))} != input_bindings.json head")
    require(complete.get("spec_sha256") == _sha(data["spec"]),
            "registry assertion: COMPLETE.json spec_sha256 != declared spec bytes")
    plan_heads = controls["head_plan"].get("heads")
    require(type(plan_heads) is list and sum(
        type(row) is dict and row.get("held_out_fold") == fold and row.get("budget_percent") == budget
        for row in plan_heads) == 1, f"registry assertion: {name_of((fold, budget))} not once in head_plan.json")
    # Registry assertion 2: head.pkl hash == accepted value == COMPLETE output binding.
    outputs = complete.get("outputs")
    cpu.keys(outputs, reader._OUTPUTS, "producer outputs")
    _digest(accepted_sha256, "accepted head.pkl sha256")
    model_sha = _sha(data["model"])
    require(model_sha == accepted_sha256,
            f"registry assertion: head.pkl sha256 {model_sha} != accepted {accepted_sha256}")
    for leaf, role in CONSUMED_OUTPUTS.items():
        entry = outputs[leaf]
        require(type(entry) is dict and entry.get("size") == len(data[role]) and
                entry.get("sha256") == _sha(data[role]), f"COMPLETE.json output binding mismatch: {leaf}")
    identity = _bank_identity(controls, bank)
    ids = types.SimpleNamespace(producer_spec_path=producer_spec_path)
    run_id = reader._producer(controls, data, ids, fold, budget)
    fingerprint = reader._plan_and_diagnostics(controls, fold, budget)
    runtime = fit._runtime_profile()
    historical = controls["diagnostics"].get("runtime")
    require(type(historical) is dict, "historical fit runtime required")
    reader._same(historical.get("distribution_versions"), runtime["distribution_versions"],
                 "historical/current accepted distribution versions")
    encoded = data["model"]
    reader._protocol4(encoded)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = legacy._pickle(encoded, "registered saved annotation head")
        except BaseException as exc:
            exc.add_note(f"saved head fold={fold}, budget={budget}: no retry/refit")
            if caught:
                exc.add_note("Model-loader warnings (not fit warnings): " +
                             reader._canonical(fit._warning_records(caught)).rstrip())
            raise
    if caught:
        error = RegistryError("head.pkl: unexpected model-loader warnings")
        error.add_note("Model-loader warnings (not fit warnings): " +
                       reader._canonical(fit._warning_records(caught)).rstrip())
        raise error
    require(type(model) is fit.FittedHead, "head.pkl: exact FittedHead required")
    require(type(model._diagnostic_snapshot) is bytes, "model diagnostic snapshot must be immutable bytes")
    reader._same(reader._json(model._diagnostic_snapshot, "model diagnostic snapshot"),
                 controls["diagnostics"], "model immutable diagnostic snapshot")
    reader._same(model.diagnostics, controls["diagnostics"], "model detached diagnostics")
    fit.predict_head(model, np.empty((0, width), dtype=np.float32, order="C"))
    provenance = {
        "held_out_fold": fold, "budget_percent": budget, "bank": bank, "encoder": bank,
        "head_bank_identity": identity, "run_id": run_id,
        "artifact_sha256": {role: _sha(data[role]) for role in (*HEAD_ASSETS, "head_plan")},
        "accepted_head_sha256": accepted_sha256, "accepted_hash_source": ACCEPTED_SOURCES[bank],
        "producer_spec_path": producer_spec_path, "fitted_state_sha256": fingerprint,
        "diagnostics_json": reader._canonical(controls["diagnostics"]),
        "consumer_runtime_json": reader._canonical(runtime), "input_width": width,
    }
    return model, provenance


def _descriptor(path, limit):
    path = Path(path)
    require(path.is_absolute(), f"{path}: absolute path required")
    size = path.stat().st_size
    require(size <= limit, f"{path}: exceeds {limit} byte limit")
    data = path.read_bytes()
    require(len(data) == size, f"{path}: changed while read")
    return data, {"path": str(path), **_binding(data)}


def _spec_location(bindings_data):
    bindings = _json(bindings_data, "input_bindings")
    entries = bindings.get("bindings")
    require(type(entries) is list, "input_bindings: bindings list required")
    specs = [entry for entry in entries if type(entry) is dict and entry.get("asset_id") == "spec"]
    require(len(specs) == 1 and specs[0].get("role") == "spec", "input_bindings: one spec binding required")
    entry = specs[0]
    for name in ("path", "resolved_path"):
        require(type(entry.get(name)) is str and entry[name].startswith("/"), f"spec binding {name}")
    return entry


def parse_acceptance_registry(data):
    """Optional H-Optimus-1 M-0 cross-check: reconciliation output_hashes by historical path."""
    value = _json(data, "acceptance registry")
    hashes = value.get("output_hashes")
    heads = value.get("heads")
    require(type(hashes) is dict and type(heads) is dict, "acceptance registry: output_hashes/heads required")
    return heads, hashes


def build_registry(bank, head_dirs, head_plan_path, *, accepted_path=None, acceptance_registry_path=None):
    require(bank in BANKS, f"unknown bank {bank!r}")
    require(len(head_dirs) == 10, "exactly ten head directories required")
    require((accepted_path is not None) == (bank == "uni2"),
            "UNI2 mode requires --accepted-heads; H-Optimus-1 mode takes the head's own COMPLETE.json")
    require(acceptance_registry_path is None or bank == "h_optimus_1",
            "--acceptance-registry is an H-Optimus-1 (M-0) cross-check only")
    plan_data, plan_descriptor = _descriptor(Path(head_plan_path).absolute(), PLAN_LIMIT)
    accepted, accepted_descriptor = None, None
    if accepted_path is not None:
        accepted_data, accepted_descriptor = _descriptor(Path(accepted_path).absolute(), REGISTRY_LIMIT)
        accepted = parse_accepted(accepted_data, bank)
    acceptance, acceptance_descriptor = None, None
    if acceptance_registry_path is not None:
        acceptance_data, acceptance_descriptor = _descriptor(Path(acceptance_registry_path).absolute(),
                                                             4 * MIB)
        acceptance = parse_acceptance_registry(acceptance_data)
    entries, seen_dirs = {}, set()
    for directory in head_dirs:
        directory = Path(directory).absolute()
        require(directory.is_dir(), f"{directory}: head directory required")
        real = directory.resolve(strict=True)
        require(real not in seen_dirs, f"{directory}: duplicate head directory")
        seen_dirs.add(real)
        data, artifacts = {"head_plan": plan_data}, {}
        for role, leaf in LEAVES.items():
            data[role], artifacts[role] = _descriptor(directory / leaf, HEAD_LIMIT)
        spec_entry = _spec_location(data["input_bindings"])
        data["spec"], artifacts["spec"] = _descriptor(Path(spec_entry["resolved_path"]), HEAD_LIMIT)
        require(artifacts["spec"]["sha256"] == spec_entry.get("sha256") and
                artifacts["spec"]["size"] == spec_entry.get("size"), f"{directory}: spec binding mismatch")
        policy = _json(data["summary"], "summary").get("policy")
        require(type(policy) is dict, f"{directory}: summary policy required")
        key = _tuple(policy.get("held_out_fold"), policy.get("budget_percent"), f"{directory} summary")
        name = name_of(key)
        require(name not in entries, f"duplicate head tuple {name}")
        complete = _json(data["complete"], "COMPLETE")
        outputs = complete.get("outputs")
        require(type(outputs) is dict and type(outputs.get("head.pkl")) is dict,
                f"{directory}: COMPLETE.json head.pkl binding required")
        if bank == "uni2":
            accepted_sha = accepted[name]
        else:
            accepted_sha = _digest(outputs["head.pkl"].get("sha256"), f"{name} COMPLETE head.pkl")
        acceptance_check = None
        if acceptance is not None:
            heads, hashes = acceptance
            require(type(heads.get(name)) is dict and
                    heads[name].get("complete_sha256") == artifacts["complete"]["sha256"],
                    f"{name}: acceptance registry COMPLETE sha256 mismatch")
            model_key = str(directory / "head.pkl")
            require(hashes.get(model_key) == accepted_sha,
                    f"{name}: acceptance registry head.pkl sha256 mismatch")
            acceptance_check = {"complete_sha256": True, "head.pkl": model_key}
        model, provenance = verify_head(
            data, held_out_fold=key[0], budget_percent=key[1], bank=bank, accepted_sha256=accepted_sha,
            width=fit._WIDTH, producer_spec_path=spec_entry["path"])
        del model
        entries[name] = {
            "held_out_fold": key[0], "budget_percent": key[1], "directory": str(directory),
            "producer_spec_path": spec_entry["path"], "artifacts": artifacts,
            "accepted_head_sha256": accepted_sha, "run_id": provenance["run_id"],
            "fitted_state_sha256": provenance["fitted_state_sha256"],
            "head_bank_identity": provenance["head_bank_identity"],
            "acceptance_registry_check": acceptance_check,
        }
    require(set(entries) == {name_of(key) for key in TUPLES}, "registry must cover exactly the ten tuples")
    module = Path(__file__).resolve(strict=True)
    return {
        "schema_version": 1, "task": TASK, "encoder": bank, "bank": bank,
        "head_plan": plan_descriptor,
        "accepted_hash_source": {"kind": ACCEPTED_SOURCES[bank], "accepted_heads": accepted_descriptor,
                                 "acceptance_registry": acceptance_descriptor},
        "input_width": fit._WIDTH,
        "heads": {name: entries[name] for name in (name_of(key) for key in TUPLES)},
        "writer": {"path": str(module), **_binding(module.read_bytes())},
        "assertions": ["registry_tuple_equals_spec_summary_input_bindings_and_head_plan",
                       "head_pkl_sha256_equals_accepted_value_and_complete_output_binding",
                       "frozen_reader_producer_plan_diagnostic_protocol4_checks",
                       "model_loaded_without_warnings_and_empty_batch_predicted_at_input_width"],
    }


def parse_registry(data, *, bank, head_plan_data, accepted_data=None):
    """Validate a HEAD_REGISTRY.json against the authenticated plan/accepted bytes."""
    value = _json(data, "head registry")
    cpu.keys(value, {"schema_version", "task", "encoder", "bank", "head_plan", "accepted_hash_source",
                     "input_width", "heads", "writer", "assertions"}, "head registry")
    require(value["schema_version"] == 1 and value["task"] == TASK, "head registry schema/task")
    require(value["bank"] == bank == value["encoder"], f"head registry bank is not {bank}")
    require(value["input_width"] == fit._WIDTH, "head registry width")
    plan = value["head_plan"]
    require(type(plan) is dict and plan.get("size") == len(head_plan_data) and
            plan.get("sha256") == _sha(head_plan_data), "head registry head_plan binding mismatch")
    source = value["accepted_hash_source"]
    require(type(source) is dict and source.get("kind") == ACCEPTED_SOURCES[bank], "accepted hash source kind")
    accepted = None
    if bank == "uni2":
        require(accepted_data is not None, "UNI2 mode requires the G-5 accepted-heads source")
        bound = source.get("accepted_heads")
        require(type(bound) is dict and bound.get("size") == len(accepted_data) and
                bound.get("sha256") == _sha(accepted_data), "head registry accepted-heads binding mismatch")
        accepted = parse_accepted(accepted_data, bank)
    else:
        require(accepted_data is None and source.get("accepted_heads") is None,
                "H-Optimus-1 mode takes no G-5 accepted-heads source")
    heads = value["heads"]
    require(type(heads) is dict and len(heads) == len(TUPLES) and
            set(heads) == {name_of(key) for key in TUPLES},
            "head registry: exactly the ten tuples")
    for key in TUPLES:
        name = name_of(key)
        entry = heads[name]
        require(type(entry) is dict and entry.get("held_out_fold") == key[0] and
                entry.get("budget_percent") == key[1], f"registry assertion: {name} entry tuple")
        artifacts = entry.get("artifacts")
        cpu.keys(artifacts, HEAD_ASSETS, f"{name} artifacts")
        for role, item in artifacts.items():
            cpu.keys(item, {"path", "size", "sha256"}, f"{name} {role}")
            require(Path(item["path"]).is_absolute(), f"{name} {role}: absolute path")
            _digest(item["sha256"], f"{name} {role}")
        _digest(entry.get("accepted_head_sha256"), f"{name} accepted")
        require(artifacts["model"]["sha256"] == entry["accepted_head_sha256"],
                f"registry assertion: {name} head.pkl sha256 != accepted value")
        if accepted is not None:
            require(entry["accepted_head_sha256"] == accepted[name],
                    f"registry assertion: {name} accepted value != G-5 accepted heads")
        require(type(entry.get("producer_spec_path")) is str and entry["producer_spec_path"].startswith("/"),
                f"{name}: historical producer spec path")
    return heads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True, choices=BANKS)
    parser.add_argument("--head-dir", action="append", required=True, help="repeat exactly ten times")
    parser.add_argument("--head-plan", required=True)
    parser.add_argument("--accepted-heads", help="UNI2 mode: lead-written G-5 accepted-heads JSON")
    parser.add_argument("--acceptance-registry",
                        help="H-Optimus-1 mode, optional: lead reconciliation registry cross-check")
    parser.add_argument("--out", required=True, help="HEAD_REGISTRY.json path (must not exist)")
    args = parser.parse_args(argv)
    out = Path(args.out).absolute()
    require(not out.exists() and not out.is_symlink(), f"{out}: refusing to overwrite")
    require(out.parent.is_dir(), f"{out.parent}: output directory must exist")
    registry = build_registry(args.bank, args.head_dir, args.head_plan, accepted_path=args.accepted_heads,
                              acceptance_registry_path=args.acceptance_registry)
    data = cpu.json_bytes(registry)
    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    require(out.read_bytes() == data, "registry readback mismatch")
    print(json.dumps({"out": str(out), **_binding(data)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
