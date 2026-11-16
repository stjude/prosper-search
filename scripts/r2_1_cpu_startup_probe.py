"""Software-only CPU startup capture and same-host post-exit verification.

Direct execution only; top level and verify are stdlib-only. No worker invocation
or scientific task is performed. The private launcher must separately run stock
worker --help in its effective python -m context after check_effective_prosper_root,
retain capture stdout (<=16 KiB), and exclusively store those exact bytes outside
the output directory. Verify receives that file and its retained-byte digest.
Custody is cooperative and same-host, not atomic or resistant to inode reuse.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import errno
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys


MIB = 1024 * 1024
CONTROL_LIMIT = 16 * MIB
HANDOFF_LIMIT = 16 * 1024
FILE_LIMIT = 4 * 1024**3
TOTAL_LIMIT = 16 * 1024**3
FILE_COUNT = 20000
BASELINE_COUNT = 4527
FORBIDDEN_COUNT = 2614
HOST = "noderome251"
TASK = "hoptimus_cpu_startup_probe"
MOSAIC = "prosper.evaluation.r2_1_hoptimus_mosaic_cli"
GENERATED = "_remote_module_non_scriptable"
DRIVER = "/usr/lib64/libcuda.so.595.58.03"
DRIVER_BINDING = {"path": DRIVER, "size": 91320456,
                  "sha256": "1cede6a524f8c1ffce093abe48cac22eb84592cd2502d8fc0658edea7ed4f3a0"}
WORKER_SHA = "c140310405f363659c96ecf46f15235248f5e2f11347d9a655f73defac05d1da"
PINS = {
    "baseline_spec": "075e5cb01842aa27855bcd6175cea1d8e040b215d1d69cd6e871e3b1580482dd",
    "baseline_profile": "b16581a7539da64c37ad3715fef43a5e8da28da40b46ee1dd2ab4a73fdd2e58e",
    "baseline_guard": "0f70430f7200358ca15c2d6f66e00a70a6ca61e27581e77c06ba4e0172e3dfd5",
    "failure_report": "aa48ae55822cbdf6c35e35b39db25b4e24f5f559b70ba05b4bc7f58f9887deab",
    "runtime_review": "183dddc7da8c3e1fff9e558be0746abf845e77f84a0511f83c3e645d4a78991f",
}
GENERATED_EXPECTED = {
    "size": 2355, "sha256": "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8",
}
STABLE_PINS = {
    "generator": "440a619c764e4133564d7956ba060a7223e94664854b94a4a2074d095756db7e",
    "template": "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc",
    "importing_api": "55c9c44ba25a2b5edf105fbd740ceff771f937147d7d6a9d6232f05681e7eeaf",
}
TORCH_NAMESPACES = {
    "torch.ops": ("_ops.py", "torch._ops"),
    "torch.classes": ("_classes.py", "torch._classes"),
}
TORCH_NAMESPACE_SOURCES = {
    "torch": {
        "path": "/HOME/miniconda3/lib/python3.12/site-packages/torch/__init__.py",
        "size": 95190,
        "sha256": "e1b5bf4151bd246a21e3ac5318f6edfcf2e6ef0fd4f233bfb77a264698ec1319",
    },
    "torch._ops": {
        "path": "/HOME/miniconda3/lib/python3.12/site-packages/torch/_ops.py",
        "size": 55360,
        "sha256": "7df9e7998b6c2343f4ae5470a7b21d458236ba9e8d27035c14a670441308028c",
    },
    "torch._classes": {
        "path": "/HOME/miniconda3/lib/python3.12/site-packages/torch/_classes.py",
        "size": 1721,
        "sha256": "2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf",
    },
}
FORMAT_FIELDS = {
    "arg_types": "*args, **kwargs", "args": "*args", "kwargs": "**kwargs",
    "arrow_and_future_return_type": "", "arrow_and_return_type": "",
    "assign_module_interface_cls": "module_interface_cls = None", "jit_script_decorator": "",
}
TERMS = ("_TEMPLATE_PREFIX", "_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
DIRECT = {
    "runner": MOSAIC, "head_reader": "prosper.evaluation.r2_1_hoptimus_head_reader",
    "fit": "prosper.evaluation.r2_1_hoptimus_fit",
    "asset_io": "prosper.evaluation.r2_1_hoptimus_asset_io",
    "head_cli": "prosper.evaluation.r2_1_hoptimus_head_cli",
    "filter_kernel": "prosper.evaluation.r2_1_hoptimus_filter",
    "filter_cli": "prosper.evaluation.r2_1_hoptimus_filter_cli",
    "cpu_io": "prosper.evaluation.r2_1_hoptimus_io", "legacy_io": "prosper.evaluation.feature_bank_audit",
    "package_init": "prosper", "evaluation_init": "prosper.evaluation",
}
INHERITED = (
    "prosper.config", "prosper.pipeline", "prosper.embeddings",
    "prosper.embeddings.patch_embedding", "prosper.embeddings.patch_embedding.base",
    "prosper.embeddings.patch_embedding.gigapath_embedding",
    "prosper.embeddings.patch_embedding.uni2_embedding",
    "prosper.embeddings.patch_embedding.virchow2_embeddings",
    "prosper.embeddings.slide_embedding", "prosper.embeddings.slide_embedding.base",
    "prosper.embeddings.slide_embedding.bob_embedding", "prosper.embeddings.slide_embedding.ot_embedding",
    "prosper.embeddings.slide_embedding.percentile_embedding", "prosper.evaluation.metrics",
    "prosper.evaluation.paths", "prosper.evaluation.slide_headers", "prosper.preprocessing",
    "prosper.preprocessing.mosaic", "prosper.search", "prosper.search.roi_search", "prosper.search.slide_search",
)
ROLE_MODULES = {**DIRECT, **{"inherited:" + name: name for name in INHERITED}}
PROFILE_TASK = "hoptimus_cpu_startup_capture_profile"
CAPTURE_NAMES = ("capture_profile.json", "persistent_bindings.sha256")
FINAL_NAME = "post_exit_verification.json"
RECORD_KEYS = ("path", "size", "sha256", "st_dev", "st_ino")
PROFILE_KEYS = (
    "schema_version", "task", "run_id", "spec_sha256", "hostname", "capture_pid", "inputs",
    "python", "cpu", "native_environment", "runtime_profile", "modules", "bootstrap_modules",
    "probe_source", "first_party", "executable_mappings", "software_bindings",
    "non_file_namespaces", "consumed_paths",
    "generated_module", "excluded_driver", "persistent_delta", "limitations",
)
CAPTURE_LIMITATIONS = [
    "software_startup_only_not_scoring_or_scientific_acceptance",
    "same_host_cooperative_custody_not_atomic_or_inode_reuse_proof",
    "observed_startup_not_future_lazy_import_closure",
    "bootstrap_modules_are_not_all_probe_only",
]


class ProbeError(ValueError):
    """A software, import-context, capture or handoff custody check failed."""


def require(condition, message):
    if not condition:
        raise ProbeError(message)


def text(value, context):
    require(type(value) is str and bool(value.strip()) and not any(c in value for c in "\n\r\x00\\"),
            f"{context}: unambiguous string required")
    return value


def integer(value, context, minimum=0):
    require(type(value) is int and value >= minimum, f"{context}: strict integer required")
    return value


def digest(value, context):
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"{context}: lowercase SHA256 required")
    return value


def absolute(value, context):
    path = Path(text(value, context))
    require(path.is_absolute(), f"{context}: absolute path required")
    return path


def keys(value, expected, context):
    require(type(value) is dict and set(value) == set(expected), f"{context}: missing/unknown keys")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(data):
    def constant(value):
        raise ProbeError(f"nonfinite JSON constant: {value}")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProbeError(f"invalid strict JSON: {exc}") from exc
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is float:
            require(math.isfinite(item), "nonfinite JSON numeric overflow")
        elif type(item) is dict:
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
    return value


def encode(value, limit=CONTROL_LIMIT):
    data = (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")
    require(len(data) <= limit, "JSON output byte limit")
    return data


def same(actual, expected, context):
    require(encode(actual) == encode(expected), f"{context}: type-sensitive mismatch")


def version(value, context):
    require(type(value) is dict and type(value.get("schema_version")) is int
            and value["schema_version"] == 1, f"{context}: integer schema_version 1 required")


def descriptor(value, context, limit=FILE_LIMIT):
    keys(value, ("path", "size", "sha256"), context)
    absolute(value["path"], context)
    require(integer(value["size"], context) <= limit, f"{context}: declared read limit")
    digest(value["sha256"], context)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(observed):
    return observed.st_dev, observed.st_ino, observed.st_size


def close(fd, primary=None):
    try:
        os.close(fd)
    except BaseException as extra:
        if primary is None:
            raise
        primary.add_note(f"descriptor cleanup: {type(extra).__name__}: {extra}")


class Inventory:
    """Distinct streamed software inventory and same-run identity observations."""

    def __init__(self, forbidden=()):
        self.forbidden_paths = {str(absolute(p, "payload").resolve()) for p in forbidden}
        self.forbidden_ids = set()
        for p in forbidden:
            try:
                observed = Path(p).stat()
            except FileNotFoundError:
                continue
            self.forbidden_ids.add((observed.st_dev, observed.st_ino))
        self.records = {}
        self.consumed_paths = {}
        self.path_limits = {}
        self.total = 0

    def path(self, value):
        path = absolute(str(value), "software path")
        require(path.suffix.lower() not in (".pkl", ".npy", ".pt", ".pth", ".safetensors"),
                "known payload suffix forbidden")
        resolved = path.resolve(strict=True)
        require(resolved.suffix.lower() not in (".pkl", ".npy", ".pt", ".pth", ".safetensors")
                and str(resolved) not in self.forbidden_paths, "payload path/alias forbidden")
        observed = path.stat()
        require((observed.st_dev, observed.st_ino) not in self.forbidden_ids, "physical payload alias forbidden")
        require(stat.S_ISREG(observed.st_mode), "regular software/control file required")
        return path, resolved, observed

    def read(self, value, limit=FILE_LIMIT, *, expected=None, retain=False):
        path, resolved, before = self.path(value)
        key = str(resolved)
        supplied = str(path)
        original = self.consumed_paths.get(supplied)
        if original is not None:
            require(original["path"] == key
                    and (original["st_dev"], original["st_ino"], original["size"]) == identity(before),
                    "declared pathname resolution/identity changed")
        else:
            require(len(self.consumed_paths) < FILE_COUNT, "declared pathname inventory limit")
        require(before.st_size <= limit, "stat read limit exceeded")
        if expected is not None:
            require(before.st_size == expected["size"], "declared file size mismatch")
        prior = self.records.get(key)
        if prior is None:
            require(len(self.records) < FILE_COUNT and self.total + before.st_size <= TOTAL_LIMIT,
                    "aggregate software inventory limit")
        else:
            require((prior["st_dev"], prior["st_ino"], prior["size"]) == identity(before),
                    "same-run software identity changed")
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        primary = None
        try:
            opened = os.fstat(fd)
            require(stat.S_ISREG(opened.st_mode) and identity(opened) == identity(before),
                    "input changed at acquisition")
            chunks = [] if retain else None
            hasher, count = hashlib.sha256(), 0
            while True:
                chunk = os.read(fd, min(MIB, min(limit, before.st_size) - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                require(count <= limit and count <= before.st_size, "running read limit/growth")
                hasher.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = path.stat()
            require(count == before.st_size and identity(os.fstat(fd)) == identity(before) == identity(after)
                    and path.resolve(strict=True) == resolved, "input truncated/replaced/resolution changed")
            record = dict(path=key, size=count, sha256=hasher.hexdigest(),
                          st_dev=after.st_dev, st_ino=after.st_ino)
            if expected is not None:
                require(record["sha256"] == expected["sha256"], "declared content digest mismatch")
            if prior is not None:
                require(record == prior, "same-run software content changed")
            else:
                self.total += count
            self.records[key] = record
            self.consumed_paths[supplied] = record
            self.path_limits[supplied] = min(limit, self.path_limits.get(supplied, limit))
            return (b"".join(chunks) if chunks is not None else None), record
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close(fd, primary)

    def sweep(self):
        for path, record in tuple(self.consumed_paths.items()):
            self.read(path, self.path_limits[path], expected=record)


def reject_cached_prosper():
    require(not any(name == "prosper" or name.startswith("prosper.") for name in sys.modules),
            "fresh interpreter required: cached prosper state")


def check_effective_prosper_root(package_init):
    """Stdlib-only helper for direct capture AND worker python -m help preflight.

    The caller must use the actual effective sys.path/cwd/environment; this
    function does not repair them and does not invoke a worker or initializer.
    """
    reject_cached_prosper()
    expected = absolute(str(package_init), "accepted package initializer").resolve(strict=True)
    found = importlib.util.find_spec("prosper")
    require(found is not None and found.origin is not None
            and Path(found.origin).resolve(strict=True) == expected,
            "effective prosper root origin mismatch before initializer")
    locations = found.submodule_search_locations
    require(locations is not None and len(locations) == 1
            and Path(locations[0]).resolve(strict=True) == expected.parent,
            "effective prosper package search location mismatch")
    return {"origin": str(expected), "search_locations": [str(expected.parent)]}


def environment(root, package_root):
    require(socket.gethostname() == HOST, "prescribed CPU host mismatch")
    require(Path(os.environ.get("PROSPER_OUT", "")).is_absolute()
            and Path(os.environ["PROSPER_OUT"]).resolve(strict=True) == root,
            "revision-local PROSPER_OUT/output_root mismatch")
    for name in THREADS:
        require(os.environ.get(name) == "1", f"{name}: native-one environment required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
            and os.environ.get("PYTHONHASHSEED") == "0", "GPU/bytecode/hashseed environment mismatch")
    require("PROSPER_PARAMS" not in os.environ, "PROSPER_PARAMS must be unset")
    if "PROSPER_MODULE" in os.environ:
        require(os.environ["PROSPER_MODULE"] == MOSAIC, "fixed worker module required")
    if "PROSPER_ROOT" in os.environ:
        require(Path(os.environ["PROSPER_ROOT"]).resolve(strict=True) == package_root,
                "worker PROSPER_ROOT/effective package mismatch")
    require(Path(sys.executable).resolve(strict=True) ==
            Path("/HOME/miniconda3/bin/python3").resolve(strict=True),
            "established interpreter required")
    supplied = os.environ.get("PYTHONPATH", "")
    require(bool(supplied) and ":" not in supplied
            and Path(supplied).resolve(strict=True) == package_root, "package-only PYTHONPATH required")
    return {name: os.environ.get(name) for name in
            (*THREADS, "CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
             "PROSPER_PARAMS", "PROSPER_OUT", "PYTHONPATH", "PROSPER_ROOT", "PROSPER_MODULE")}


def driver_absent():
    require(not Path(DRIVER).exists() and not Path(DRIVER).is_symlink(),
            "recorded driver is no longer absent")


def checksum_list(data):
    require(len(data) <= CONTROL_LIMIT, "checksum list limit")
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ProbeError(f"invalid checksum UTF8: {exc}") from exc
    result = {}
    require(bool(lines) and data.endswith(b"\n"), "complete checksum lines required")
    for line in lines:
        require(len(line) > 66 and line[64:66] == "  ", "ambiguous checksum line")
        value, path = line[:64], line[66:]
        digest(value, "checksum")
        absolute(path, "checksum path")
        require(path not in result, "duplicate checksum path")
        result[path] = value
    return result


def record_schema(record, context):
    keys(record, RECORD_KEYS, context)
    absolute(record["path"], context)
    integer(record["size"], context)
    digest(record["sha256"], context)
    integer(record["st_dev"], context)
    integer(record["st_ino"], context, 1)


def prepare(spec_path, spec_digest, out):
    reject_cached_prosper()
    digest(spec_digest, "spec digest")
    inv = Inventory()
    raw, spec_record = inv.read(absolute(str(spec_path), "spec"), CONTROL_LIMIT, retain=True)
    require(spec_record["sha256"] == spec_digest, "probe spec digest mismatch")
    spec = strict_json(raw)
    keys(spec, ("schema_version", "run_id", "task", "expected_host", "sources", "code_files",
                "output_root"), "probe spec")
    version(spec, "probe spec")
    text(spec["run_id"], "run_id")
    require(spec["task"] == TASK and spec["expected_host"] == HOST, "probe task/host mismatch")
    keys(spec["sources"], PINS, "source roles")
    keys(spec["code_files"], ("probe", "worker"), "code roles")
    root = absolute(spec["output_root"], "output_root").resolve(strict=True)
    output = absolute(str(out), "output")
    canonical_out = output.parent.resolve(strict=True) / output.name
    require(root != canonical_out and root in canonical_out.parents, "output must be inside output_root")
    for role, entry in spec["sources"].items():
        descriptor(entry, role, CONTROL_LIMIT)
        require(entry["sha256"] == PINS[role], f"{role}: frozen baseline pin mismatch")
    for role, entry in spec["code_files"].items():
        descriptor(entry, role, CONTROL_LIMIT)
    require(Path(spec["code_files"]["probe"]["path"]).resolve(strict=True) == Path(__file__).resolve(),
            "actual directly executing probe source mismatch")
    require(spec["code_files"]["worker"]["sha256"] == WORKER_SHA, "frozen worker pin mismatch")
    source = spec["sources"]["baseline_spec"]
    baseline_bytes, _ = inv.read(source["path"], CONTROL_LIMIT, expected=source, retain=True)
    baseline = strict_json(baseline_bytes)
    version(baseline, "baseline spec")
    require(baseline.get("task") == "hoptimus_mosaic_predictions", "baseline task mismatch")
    first = baseline.get("code_files")
    keys(first, ROLE_MODULES, "32 first-party roles")
    require(type(baseline.get("features")) is list and type(baseline.get("heads")) is list,
            "baseline feature/head metadata lists required")
    forbidden = []
    for entry in baseline["features"]:
        require(type(entry) is dict and "path" in entry, "feature payload path metadata required")
        forbidden.append(str(absolute(entry["path"], "feature payload")))
    for head in baseline["heads"]:
        require(type(head) is dict and type(head.get("assets")) is dict
                and type(head["assets"].get("model")) is dict
                and "path" in head["assets"]["model"], "head model payload path metadata required")
        forbidden.append(str(absolute(head["assets"]["model"]["path"], "head payload")))
    require(len(forbidden) == FORBIDDEN_COUNT and len(set(forbidden)) == FORBIDDEN_COUNT,
            "exact forbidden payload metadata required")
    exclusions = Inventory(forbidden)
    inv.forbidden_paths = exclusions.forbidden_paths
    inv.forbidden_ids = exclusions.forbidden_ids
    for role, entry in first.items():
        descriptor(entry, role, CONTROL_LIMIT)
    env = environment(root, Path(first["package_init"]["path"]).resolve(strict=True).parent.parent)
    controls = {}
    inputs = {"spec": spec_record}
    for role, entry in spec["sources"].items():
        payload, observed = inv.read(entry["path"], CONTROL_LIMIT, expected=entry, retain=True)
        controls[role] = payload
        inputs[role] = observed
    for role, entry in spec["code_files"].items():
        _, inputs[role] = inv.read(entry["path"], CONTROL_LIMIT, expected=entry)
    profile = strict_json(controls["baseline_profile"])
    review = strict_json(controls["runtime_review"])
    failure = strict_json(controls["failure_report"])
    for name, item in (("baseline profile", profile), ("runtime review", review), ("failure", failure)):
        version(item, name)
    require(review.get("disposition") == "accepted"
            and review.get("review_kind") == "exact_amended_mosaic_runtime_resource_declaration"
            and review.get("spec_sha256") == PINS["baseline_spec"]
            and review.get("runtime_profile_sha256") == PINS["baseline_profile"]
            and review.get("launch_sha256") == PINS["baseline_guard"], "baseline runtime review links mismatch")
    same(failure.get("failed_guard"), DRIVER_BINDING, "recorded failed driver")
    require(failure.get("worker_host") == HOST
            and type(failure.get("guard_entries_passed")) is int
            and failure["guard_entries_passed"] == BASELINE_COUNT - 1
            and failure.get("worker_entry_observed") is False, "recorded sole-driver failure mismatch")
    same(profile.get("first_party_code"), first, "baseline first-party metadata")
    guard = checksum_list(controls["baseline_guard"])
    require(len(guard) == BASELINE_COUNT and guard.get(DRIVER) == DRIVER_BINDING["sha256"],
            "baseline launch guard/sole driver mismatch")
    same(profile["software_bindings"].get(DRIVER), DRIVER_BINDING, "baseline driver observation")
    driver_absent()
    generation = profile["generated_module_bindings"][GENERATED]
    require(generation["historical_observation"]["path"] not in guard,
            "expired generated source must not be persistent")
    for path, expected in guard.items():
        if path == DRIVER:
            continue
        known = profile["software_bindings"].get(path)
        if known is not None:
            descriptor(known, "baseline software")
            require(known["sha256"] == expected, "baseline software/guard mismatch")
        _, observed = inv.read(path, expected=known)
        require(observed["sha256"] == expected, "original launch guard mismatch")
    for role, entry in first.items():
        inv.read(entry["path"], CONTROL_LIMIT, expected=entry)
    return dict(spec=spec, spec_digest=spec_digest, root=root, out=canonical_out,
                inventory=inv, controls=controls, inputs=inputs, first=first, baseline=profile,
                guard=guard, environment=env, forbidden=forbidden)


def derive_generated(context):
    declaration = context["baseline"]["generated_module_bindings"][GENERATED]
    require(declaration["kind"] == "deterministic_import_time_generated_source"
            and declaration["persistent_path_required"] is False
            and declaration["factory"] == "instantiate_non_scriptable_remote_module_template",
            "generated-source binding kind/factory mismatch")
    same(declaration["expected_source"], GENERATED_EXPECTED, "generated-source expected bytes")
    same(declaration["fixed_inputs"], dict(
                generated_module_name=GENERATED, enable_moving_cpu_tensors_to_cuda=True,
                format_fields=FORMAT_FIELDS), "generated-source fixed inputs")
    same(declaration["template_terms_concatenated"], list(TERMS), "generated-source template terms")
    texts = {}
    keys(declaration["stable_sources"], STABLE_PINS, "three generated-source bindings")
    for role, entry in declaration["stable_sources"].items():
        descriptor(entry, role, CONTROL_LIMIT)
        require(entry["sha256"] == STABLE_PINS[role], "pinned generator/API/template mismatch")
        texts[role], _ = context["inventory"].read(entry["path"], CONTROL_LIMIT, expected=entry, retain=True)
    tree = ast.parse(texts["generator"])
    assignments = {node.targets[0].id: node.value for node in tree.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)}
    require(ast.literal_eval(assignments["_FILE_PREFIX"]) == "_remote_module_", "generator prefix mismatch")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == declaration["factory"]]
    require(len(functions) == 1, "one fixed generator factory required")
    factory = functions[0]
    values = [node.value for node in factory.body if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == "str_dict"]
    require(len(values) == 1 and isinstance(values[0], ast.Call)
            and isinstance(values[0].func, ast.Name) and values[0].func.id == "dict"
            and not values[0].args, "literal factory fields required")
    fields = {keyword.arg: ast.literal_eval(keyword.value) for keyword in values[0].keywords}
    require(len(fields) == len(values[0].keywords) and fields == FORMAT_FIELDS,
            "actual generator fixed fields mismatch")
    last = factory.body[-1]
    require(isinstance(last, ast.Return) and isinstance(last.value, ast.Call), "generator return call required")
    call = last.value
    require(isinstance(call.func, ast.Name) and call.func.id == "_do_instantiate_remote_module_template"
            and not call.keywords and len(call.args) == 3
            and all(isinstance(a, ast.Name) for a in call.args[:2])
            and [a.id for a in call.args[:2]] == ["generated_module_name", "str_dict"]
            and isinstance(call.args[2], ast.Constant) and call.args[2].value is True,
            "generator fixed invocation mismatch")
    require(texts["importing_api"].count(b"instantiator.instantiate_non_scriptable_remote_module_template()") == 1,
            "fixed importing API invocation mismatch")
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in ast.parse(texts["template"]).body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)}
    generated = "".join(constants[name] for name in TERMS).format(**fields).encode("utf-8")
    require({"size": len(generated), "sha256": sha(generated)} == GENERATED_EXPECTED,
            "literal generated source derivation mismatch")
    return declaration


def _runtime():
    module = importlib.import_module(MOSAIC)
    deferred = module._reader()
    paths = module.code_paths(deferred)
    fitter = sys.modules["prosper.evaluation.r2_1_hoptimus_fit"]
    return paths, fitter._runtime_profile()


def runtime_schema(runtime):
    keys(runtime, ("distribution_versions", "implementation", "native_pools"), "frozen runtime getter")
    require(type(runtime["distribution_versions"]) is dict, "runtime version mapping required")
    for name, value in runtime["distribution_versions"].items():
        text(name, "runtime distribution")
        text(value, "runtime distribution version")
    keys(runtime["implementation"], ("sklearn", "numpy", "scipy", "SGDClassifier"),
         "runtime implementation")
    for name, item in runtime["implementation"].items():
        keys(item, ("path",) if name == "SGDClassifier" else ("version", "path"),
             "runtime implementation entry")
        absolute(item["path"], "runtime implementation source")
        if name != "SGDClassifier":
            text(item["version"], "runtime implementation version")
    require(type(runtime["native_pools"]) is list, "runtime native pools required")
    for pool in runtime["native_pools"]:
        require(type(pool) is dict and type(pool.get("num_threads")) is int
                and pool["num_threads"] == 1, "non-unit native pools")


def module_schema(modules):
    require(type(modules) is dict, "loaded-module mapping required")
    for name, item in modules.items():
        text(name, "module key")
        keys(item, ("name", "file", "origin", "search_locations", "phase"), "module record")
        text(item["name"], "module name")
        if name in TORCH_NAMESPACES:
            marker, _ = TORCH_NAMESPACES[name]
            require(item["name"] == name and item["file"] == marker
                    and item["origin"] is None and item["search_locations"] is None,
                    "pinned non-file namespace metadata mismatch")
        elif item["file"] is not None:
            absolute(item["file"], "module source")
        if item["origin"] is not None:
            text(item["origin"], "module origin")
        if item["search_locations"] is not None:
            require(type(item["search_locations"]) is list, "module search location list required")
            for path in item["search_locations"]:
                absolute(path, "module search location")
        require(item["phase"] in ("bootstrap", "startup"), "module phase malformed")


def namespace_provenance(context, modules):
    result = {}
    for name, (marker, defining) in TORCH_NAMESPACES.items():
        if name not in modules:
            continue
        sources = {}
        for source_name in ("torch", defining):
            require(source_name in modules, "non-file namespace defining module missing")
            module = modules[source_name]
            expected = TORCH_NAMESPACE_SOURCES[source_name]
            location = Path(expected["path"]).resolve(strict=True)
            require(module["name"] == source_name
                    and module["file"] is not None and module["origin"] is not None
                    and Path(module["file"]).resolve(strict=True) == location
                    and Path(module["origin"]).resolve(strict=True) == location,
                    "non-file namespace defining-source origin mismatch")
            _, sources[source_name] = context["inventory"].read(
                expected["path"], CONTROL_LIMIT, expected=expected)
        result[name] = dict(reported_file=marker, defining_module=defining, sources=sources)
    return result


def _module_records(bootstrap):
    records = {}
    for name, module in tuple(sys.modules.items()):
        require(module is not None, f"missing loaded module object: {name}")
        spec = getattr(module, "__spec__", None)
        locations = getattr(spec, "submodule_search_locations", None)
        records[name] = dict(
            name=getattr(module, "__name__", None),
            file=getattr(module, "__file__", None), origin=getattr(spec, "origin", None),
            search_locations=None if locations is None else list(locations),
            phase="bootstrap" if name in bootstrap else "startup",
        )
    return records


def _mapped_paths():
    fd = os.open("/proc/self/maps", os.O_RDONLY)
    primary = None
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "regular proc maps diagnostic required")
        chunks, count = [], 0
        while True:
            chunk = os.read(fd, min(MIB, CONTROL_LIMIT - count + 1))
            if not chunk:
                break
            count += len(chunk)
            require(count <= CONTROL_LIMIT, "proc maps diagnostic running limit")
            chunks.append(chunk)
        require(identity(os.fstat(fd)) == identity(before)
                == identity(Path("/proc/self/maps").stat()), "proc maps identity changed")
        result = set()
        for line in b"".join(chunks).decode("utf-8").splitlines():
            fields = line.split(None, 5)
            if len(fields) == 6 and "x" in fields[1] and fields[5].startswith("/"):
                require(not fields[5].endswith(" (deleted)"), "deleted executable mapping")
                result.add(fields[5])
        return sorted(result)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close(fd, primary)


def reconcile_first_party(context, modules, paths=None):
    expected = {name: context["first"][role] for role, name in ROLE_MODULES.items()}
    names = {name for name in modules if name == "prosper" or name.startswith("prosper.")}
    require(names == set(expected), "actual first-party module inventory mismatch")
    if paths is not None:
        require(set(paths) == set(ROLE_MODULES), "actual code_paths roles mismatch")
        for role, path in paths.items():
            require(Path(path).resolve(strict=True) == Path(context["first"][role]["path"]).resolve(strict=True),
                    "actual code_paths origin mismatch")
    result = {}
    for name, entry in expected.items():
        module = modules[name]
        location = Path(entry["path"]).resolve(strict=True)
        require(module["name"] == name and module["file"] is not None and module["origin"] is not None
                and Path(module["file"]).resolve(strict=True) == location
                and Path(module["origin"]).resolve(strict=True) == location,
                f"{name}: actual file/spec origin mismatch")
        locations = module["search_locations"]
        if location.name == "__init__.py":
            require(type(locations) is list and len(locations) == 1
                    and Path(locations[0]).resolve(strict=True) == location.parent,
                    f"{name}: package search locations mismatch")
        else:
            require(locations is None, f"{name}: unexpected package search locations")
        _, result[name] = context["inventory"].read(location, CONTROL_LIMIT, expected=entry)
    return result


class _NoStdout:
    def write(self, value):
        require(value == "", "unexpected startup stdout; capture handoff cannot filter it")
        return 0

    def flush(self):
        pass


@contextmanager
def no_startup_stdout():
    original = sys.stdout
    sys.stdout = _NoStdout()
    try:
        yield
    finally:
        sys.stdout = original


class OwnedDirectory:
    def __init__(self, path, *, create=False, expected=None):
        self.path = Path(path)
        if create:
            self.path.mkdir()
        before = self.path.lstat()
        require(stat.S_ISDIR(before.st_mode), "owned output directory required, not symlink")
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.device, self.inode = before.st_dev, before.st_ino
        self.records = {}
        try:
            self.check()
            if expected is not None:
                require(expected == self.directory_record(), "capture directory identity mismatch")
        except BaseException as exc:
            close(self.fd, exc)
            raise

    def directory_record(self):
        return dict(path=str(self.path), st_dev=self.device, st_ino=self.inode)

    def release(self, primary=None):
        try:
            opened = os.fstat(self.fd)
            require(stat.S_ISDIR(opened.st_mode)
                    and (opened.st_dev, opened.st_ino) == (self.device, self.inode),
                    "owned directory descriptor changed before release")
        except BaseException as extra:
            if primary is None:
                raise
            primary.add_note(f"owned directory descriptor release: {extra}")
            return
        close(self.fd, primary)

    def check(self):
        observed, opened = self.path.lstat(), os.fstat(self.fd)
        require(stat.S_ISDIR(observed.st_mode) and stat.S_ISDIR(opened.st_mode)
                and (observed.st_dev, observed.st_ino) == (opened.st_dev, opened.st_ino)
                == (self.device, self.inode) and self.path.resolve(strict=True) == self.path,
                "output directory custody changed")

    def names(self):
        self.check()
        return set(os.listdir(self.fd))

    def write(self, name, data):
        self.check()
        require(len(data) <= CONTROL_LIMIT, "owned output byte limit")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        primary = None
        try:
            observed = os.fstat(fd)
            self.records[name] = dict(size=0, sha256=sha(b""), st_dev=observed.st_dev, st_ino=observed.st_ino)
            view, count = memoryview(data), 0
            while count < len(view):
                wrote = os.write(fd, view[count:count + MIB])
                require(wrote > 0, "short owned write")
                count += wrote
            os.fsync(fd)
            require(os.fstat(fd).st_size == len(data), "owned output write size")
            self.records[name] = dict(size=len(data), sha256=sha(data),
                                      st_dev=observed.st_dev, st_ino=observed.st_ino)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close(fd, primary)
        return self.read(name, self.records[name])

    def read(self, name, expected):
        self.check()
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        primary = None
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode)
                    and identity(before) == (expected["st_dev"], expected["st_ino"], expected["size"])
                    and before.st_size <= CONTROL_LIMIT, "owned artifact identity/size mismatch")
            chunks, count = [], 0
            while True:
                chunk = os.read(fd, min(MIB, before.st_size - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                require(count <= before.st_size and count <= CONTROL_LIMIT, "owned artifact running limit/growth")
                chunks.append(chunk)
            after = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            data = b"".join(chunks)
            require(stat.S_ISREG(after.st_mode) and identity(os.fstat(fd)) == identity(before) == identity(after)
                    and count == before.st_size and sha(data) == expected["sha256"],
                    "owned artifact content/custody mismatch")
            self.check()
            return data
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close(fd, primary)

    def remove_owned(self, name, primary):
        record = self.records.get(name)
        if record is None:
            return
        try:
            try:
                opened = os.fstat(self.fd)
                usable = stat.S_ISDIR(opened.st_mode) and \
                    (opened.st_dev, opened.st_ino) == (self.device, self.inode)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
                usable = False
            if not usable:
                # A catchable error can follow descriptor release. Reacquire only
                # this directory; a reused foreign descriptor is not ours.
                reopened = OwnedDirectory(self.path, expected=self.directory_record())
                try:
                    reopened.records[name] = record
                    reopened.remove_owned(name, primary)
                finally:
                    reopened.release(primary)
                return
            self.check()
            try:
                observed = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            if stat.S_ISREG(observed.st_mode) and (observed.st_dev, observed.st_ino) == \
                    (record["st_dev"], record["st_ino"]):
                os.unlink(name, dir_fd=self.fd)
        except BaseException as extra:
            primary.add_note(f"owned-only verification receipt cleanup: {extra}")


def receipt_schema(receipt):
    keys(receipt, ("schema_version", "task", "run_id", "spec_sha256", "capture_pid",
                   "hostname", "output_directory", "artifacts"), "capture handoff")
    version(receipt, "capture handoff")
    require(receipt["task"] == "hoptimus_cpu_startup_capture", "handoff task mismatch")
    text(receipt["run_id"], "handoff run")
    digest(receipt["spec_sha256"], "handoff spec")
    integer(receipt["capture_pid"], "handoff PID", 1)
    require(receipt["hostname"] == HOST, "handoff prescribed host mismatch")
    directory = receipt["output_directory"]
    keys(directory, ("path", "st_dev", "st_ino"), "handoff directory")
    absolute(directory["path"], "handoff directory")
    integer(directory["st_dev"], "handoff directory device")
    integer(directory["st_ino"], "handoff directory inode", 1)
    keys(receipt["artifacts"], CAPTURE_NAMES, "handoff artifacts")
    for name, item in receipt["artifacts"].items():
        keys(item, ("size", "sha256", "st_dev", "st_ino"), name)
        integer(item["size"], name, 1)
        require(item["size"] <= CONTROL_LIMIT, "handoff artifact limit")
        digest(item["sha256"], name)
        integer(item["st_dev"], name)
        integer(item["st_ino"], name, 1)


def persistent_delta(context, software):
    baseline = context["baseline"]["software_bindings"]
    old, new = set(baseline), set(software)
    return dict(added=sorted(new - old), dropped=sorted(old - new),
                changed=[dict(path=p, before=baseline[p], after=software[p])
                         for p in sorted(old & new) if
                         baseline[p]["sha256"] != software[p]["sha256"]
                         or baseline[p]["size"] != software[p]["size"]])


def capture(spec_path, spec_digest, out):
    context = prepare(spec_path, spec_digest, out)
    check_effective_prosper_root(context["first"]["package_init"]["path"])
    derivation = derive_generated(context)
    bootstrap = sorted(sys.modules)
    owned = OwnedDirectory(context["out"], create=True)
    primary = None
    try:
        with no_startup_stdout():
            paths, runtime = _runtime()
        runtime_schema(runtime)
        modules = _module_records(bootstrap)
        module_schema(modules)
        namespaces = namespace_provenance(context, modules)
        first = reconcile_first_party(context, modules, paths)
        require(GENERATED in modules and modules[GENERATED]["file"] is not None,
                "exact generated module missing while alive")
        generated_path = absolute(modules[GENERATED]["file"], "generated source")
        require(generated_path.name == GENERATED + ".py"
                and modules[GENERATED]["origin"] is not None
                and Path(modules[GENERATED]["origin"]).resolve(strict=True) == generated_path.resolve(strict=True),
                "generated source origin/name mismatch")
        transient = Inventory(context["forbidden"])
        _, generated = transient.read(generated_path, CONTROL_LIMIT, expected=GENERATED_EXPECTED)
        require(generated["path"] not in context["inventory"].records,
                "generated source must not be persistent")
        mappings = _mapped_paths()
        for name, item in modules.items():
            if name != GENERATED and name not in namespaces and item["file"] is not None:
                context["inventory"].read(item["file"])
        for path in mappings:
            context["inventory"].read(path)
        require(generated["path"] not in context["inventory"].records, "generated source leaked into persistence")
        software = dict(sorted(context["inventory"].records.items()))
        profile = dict(
            schema_version=1, task=PROFILE_TASK, run_id=context["spec"]["run_id"],
            spec_sha256=spec_digest, hostname=socket.gethostname(), capture_pid=os.getpid(),
            inputs=context["inputs"], python=dict(executable=sys.executable, version=sys.version),
            cpu=dict(count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0))),
            native_environment=context["environment"], runtime_profile=runtime, modules=modules,
            bootstrap_modules=bootstrap, probe_source=context["inputs"]["probe"], first_party=first,
            executable_mappings=mappings, software_bindings=software,
            non_file_namespaces=namespaces, consumed_paths=dict(context["inventory"].consumed_paths),
            generated_module=dict(name=GENERATED, observed=generated, derivation=derivation),
            excluded_driver=DRIVER_BINDING, persistent_delta=persistent_delta(context, software),
            limitations=CAPTURE_LIMITATIONS,
        )
        owned.write(CAPTURE_NAMES[0], encode(profile))
        listing = "".join(f"{entry['sha256']}  {path}\n" for path, entry in software.items()).encode("utf-8")
        require(len(listing) <= CONTROL_LIMIT, "persistent checksum output limit")
        owned.write(CAPTURE_NAMES[1], listing)
        context["inventory"].sweep()
        final_modules = _module_records(bootstrap)
        module_schema(final_modules)
        reconcile_first_party(context, final_modules, paths)
        same(namespace_provenance(context, final_modules), namespaces, "final non-file namespace provenance")
        transient.sweep()
        driver_absent()
        require(owned.names() == set(CAPTURE_NAMES), "unexpected capture output files")
        for name, record in owned.records.items():
            owned.read(name, record)
        receipt = dict(schema_version=1, task="hoptimus_cpu_startup_capture",
                       run_id=context["spec"]["run_id"], spec_sha256=spec_digest,
                       capture_pid=os.getpid(), hostname=socket.gethostname(),
                       output_directory=owned.directory_record(), artifacts=dict(owned.records))
        receipt_schema(receipt)
        retained = encode(receipt, HANDOFF_LIMIT)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        owned.release(primary)
    # Closing precedes this final original-path/file check and the stdout handoff.
    final = OwnedDirectory(context["out"], expected=receipt["output_directory"])
    primary = None
    try:
        driver_absent()
        require(final.names() == set(CAPTURE_NAMES), "capture allowlist changed after close")
        for name, record in receipt["artifacts"].items():
            final.read(name, record)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        final.release(primary)
    return retained


def validate_profile(profile, context, receipt, inventory):
    keys(profile, PROFILE_KEYS, "capture profile")
    version(profile, "capture profile")
    text(profile["run_id"], "capture run")
    digest(profile["spec_sha256"], "capture spec")
    integer(profile["capture_pid"], "capture PID", 1)
    require(profile["task"] == PROFILE_TASK and profile["run_id"] == receipt["run_id"]
            and profile["spec_sha256"] == receipt["spec_sha256"]
            and profile["hostname"] == receipt["hostname"]
            and profile["capture_pid"] == receipt["capture_pid"], "capture profile linkage mismatch")
    same(profile["inputs"], context["inputs"], "capture original control/code bindings")
    same(profile["excluded_driver"], DRIVER_BINDING, "capture sole-driver exclusion")
    keys(profile["python"], ("executable", "version"), "captured interpreter")
    text(profile["python"]["version"], "captured Python version")
    require(absolute(profile["python"]["executable"], "captured interpreter").resolve(strict=True) ==
            Path(sys.executable).resolve(strict=True), "captured interpreter changed")
    keys(profile["cpu"], ("count", "affinity"), "captured CPU")
    integer(profile["cpu"]["count"], "CPU count", 1)
    require(type(profile["cpu"]["affinity"]) is list and bool(profile["cpu"]["affinity"]),
            "captured CPU affinity required")
    for cpu in profile["cpu"]["affinity"]:
        integer(cpu, "CPU affinity")
    require(len(set(profile["cpu"]["affinity"])) == len(profile["cpu"]["affinity"]),
            "captured CPU affinity duplicates")
    keys(profile["generated_module"], ("name", "observed", "derivation"), "generated-source provenance")
    require(profile["generated_module"]["name"] == GENERATED, "captured generated module name changed")
    same(profile["generated_module"]["derivation"], derive_generated(context),
         "captured generated-source derivation")
    observed = profile["generated_module"]["observed"]
    record_schema(observed, "generated observation")
    same({k: observed[k] for k in ("size", "sha256")}, GENERATED_EXPECTED, "generated bytes")
    require(Path(observed["path"]).name == GENERATED + ".py"
            and not Path(observed["path"]).exists() and not Path(observed["path"]).is_symlink(),
            "capture generated source must now be absent")
    runtime = profile["runtime_profile"]
    runtime_schema(runtime)
    same(profile["native_environment"], context["environment"], "capture environment")
    module_schema(profile["modules"])
    first = reconcile_first_party(context, profile["modules"])
    same(profile["first_party"], first, "captured first-party records")
    require(type(profile["bootstrap_modules"]) is list, "bootstrap module accounting malformed")
    for name in profile["bootstrap_modules"]:
        text(name, "bootstrap module name")
    require(len(set(profile["bootstrap_modules"])) == len(profile["bootstrap_modules"]),
            "bootstrap module accounting duplicates")
    software = profile["software_bindings"]
    require(type(software) is dict and len(software) <= FILE_COUNT
            and observed["path"] not in software and DRIVER not in software, "persistent inventory malformed")
    for path, entry in software.items():
        record_schema(entry, "captured software")
        require(path == entry["path"], "persistent path/record mismatch")
        _, actual = inventory.read(path, expected=entry)
        require(actual == entry, "same-host persistent file identity/content changed")
    same(profile["persistent_delta"], persistent_delta(context, software), "persistent delta")
    consumed = profile["consumed_paths"]
    require(type(consumed) is dict and len(consumed) <= FILE_COUNT, "consumed pathname inventory malformed")
    for path, entry in consumed.items():
        absolute(path, "consumed pathname")
        record_schema(entry, "consumed pathname target")
        require(entry["path"] in software, "consumed pathname target not persistent")
        same(entry, software[entry["path"]], "consumed pathname canonical binding")
        _, actual = inventory.read(path, expected=entry)
        same(actual, entry, "consumed pathname identity/content")
    require({entry["path"] for entry in consumed.values()} == set(software),
            "consumed pathname coverage mismatch")
    same(profile["non_file_namespaces"], namespace_provenance(context, profile["modules"]),
         "captured non-file namespace provenance")
    for name, item in profile["modules"].items():
        require(item["phase"] == ("bootstrap" if name in profile["bootstrap_modules"] else "startup"),
                "module bootstrap/startup accounting mismatch")
        if name != GENERATED and name not in TORCH_NAMESPACES and item["file"] is not None:
            require(str(Path(item["file"]).resolve(strict=True)) in software, "observed module not persistent")
    require(type(profile["executable_mappings"]) is list, "executable mapping list required")
    for path in profile["executable_mappings"]:
        absolute(path, "executable mapping")
        require(str(Path(path).resolve(strict=True)) in software, "executable mapping not persistent")
    for item in runtime["implementation"].values():
        require(str(Path(item["path"]).resolve(strict=True)) in software,
                "runtime implementation source not persistent")
    same(profile["probe_source"], context["inputs"]["probe"], "probe-only source accounting")
    same(profile["limitations"], CAPTURE_LIMITATIONS, "capture limitations")


def verify(spec_path, spec_digest, out, capture_receipt, capture_digest):
    digest(capture_digest, "trusted handoff digest")
    context = prepare(spec_path, spec_digest, out)
    handoff = absolute(str(capture_receipt), "capture receipt")
    require(context["out"] not in handoff.resolve(strict=True).parents
            and handoff.resolve(strict=True) != context["out"], "handoff must be outside diagnostic directory")
    retained, handoff_record = context["inventory"].read(handoff, HANDOFF_LIMIT, retain=True)
    require(handoff_record["sha256"] == capture_digest, "trusted capture handoff digest mismatch")
    require(retained.endswith(b"\n") and retained.count(b"\n") == 1, "exact one-line capture stdout required")
    receipt = strict_json(retained)
    receipt_schema(receipt)
    require(receipt["run_id"] == context["spec"]["run_id"] and receipt["spec_sha256"] == spec_digest
            and receipt["hostname"] == socket.gethostname() and receipt["capture_pid"] != os.getpid(),
            "handoff run/spec/host/fresh PID mismatch")
    require(receipt["output_directory"]["path"] == str(context["out"]), "handoff output path mismatch")
    owned = OwnedDirectory(context["out"], expected=receipt["output_directory"])
    primary = None
    try:
        require(owned.names() == set(CAPTURE_NAMES), "verify requires exactly two capture outputs")
        profile_data = owned.read(CAPTURE_NAMES[0], receipt["artifacts"][CAPTURE_NAMES[0]])
        listing_data = owned.read(CAPTURE_NAMES[1], receipt["artifacts"][CAPTURE_NAMES[1]])
        profile = strict_json(profile_data)
        software_inventory = Inventory(context["forbidden"])
        validate_profile(profile, context, receipt, software_inventory)
        require(checksum_list(listing_data) == {
            path: entry["sha256"] for path, entry in profile["software_bindings"].items()
        }, "persistent list/profile reconciliation mismatch")
        context["inventory"].sweep()
        software_inventory.sweep()
        driver_absent()
        for name, entry in receipt["artifacts"].items():
            owned.read(name, entry)
        result = dict(
            schema_version=1, task="hoptimus_cpu_startup_post_exit_verification",
            run_id=receipt["run_id"], spec_sha256=spec_digest, hostname=socket.gethostname(),
            capture_pid=receipt["capture_pid"], verify_pid=os.getpid(),
            capture_handoff=handoff_record, capture_artifacts=receipt["artifacts"],
            capture_handoff_path=dict(path=str(handoff), target=handoff_record),
            output_directory=receipt["output_directory"], checked_bindings=software_inventory.records,
            excluded_driver=DRIVER_BINDING, generated_source_absent=profile["generated_module"]["observed"],
            limitations=["software_diagnostic_not_runtime_or_scoring_acceptance",
                         "same_host_cooperative_custody_not_inode_reuse_proof"],
        )
        owned.write(FINAL_NAME, encode(result))
        context["inventory"].sweep()
        software_inventory.sweep()
        driver_absent()
        for name, entry in receipt["artifacts"].items():
            owned.read(name, entry)
        owned.read(FINAL_NAME, owned.records[FINAL_NAME])
        require(owned.names() == set((*CAPTURE_NAMES, FINAL_NAME)), "verify final output allowlist")
    except BaseException as exc:
        primary = exc
        owned.remove_owned(FINAL_NAME, exc)
        raise
    finally:
        try:
            owned.release(primary)
        except BaseException as release_error:
            owned.remove_owned(FINAL_NAME, release_error)
            raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("capture", "verify"))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--capture-receipt")
    parser.add_argument("--capture-receipt-sha256")
    args = parser.parse_args(argv)
    if args.mode == "capture":
        require(args.capture_receipt is None and args.capture_receipt_sha256 is None,
                "capture rejects verify-only handoff arguments")
        retained = capture(args.spec, args.spec_sha256, args.out)
        sys.stdout.write(retained.decode("utf-8"))
        sys.stdout.flush()
    else:
        require(args.capture_receipt is not None and args.capture_receipt_sha256 is not None,
                "verify requires capture receipt and retained-byte SHA256")
        verify(args.spec, args.spec_sha256, args.out, args.capture_receipt, args.capture_receipt_sha256)


if __name__ == "__main__":
    main()
