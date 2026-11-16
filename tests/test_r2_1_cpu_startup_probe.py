"""Synthetic stdlib-only startup/custody checks, never actual package startup."""

import ast
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/r2_1_cpu_startup_probe.py"
SPEC = importlib.util.spec_from_file_location("synthetic_cpu_startup_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def entry(path):
    data = path.read_bytes()
    return dict(path=str(path), size=len(data), sha256=sha(data))


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def json_file(path, value):
    return put(path, probe.encode(value))


def forbidden(*args, **kwargs):
    pytest.fail("forbidden package/model/native operation")


def module(name, path, package=False):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    obj.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, origin=str(path), is_package=package)
    if package:
        obj.__spec__.submodule_search_locations = [str(path.parent)]
    return obj


@pytest.fixture
def packet(tmp_path, monkeypatch):
    require_no_real = [name for name in sys.modules if name == "prosper" or name.startswith("prosper.")]
    assert not require_no_real, "test runner must not import actual prosper"
    root = tmp_path / "output_root"
    root.mkdir()
    package_root = tmp_path / "package"
    package_root.mkdir()
    first, modules = {}, {}
    names = set(probe.ROLE_MODULES.values())
    for role, name in probe.ROLE_MODULES.items():
        package = any(other.startswith(name + ".") for other in names)
        parts = name.split(".")
        path = package_root.joinpath(*parts, "__init__.py") if package else \
            package_root.joinpath(*parts[:-1], parts[-1] + ".py")
        put(path, b"raise AssertionError('synthetic initializer/module must not execute')\n")
        first[role] = entry(path)
        modules[name] = module(name, path, package)
    worker = put(tmp_path / "worker.sh", b"synthetic worker, never execute\n")
    wheel = put(tmp_path / "libtorch_cuda.so", b"GPU-named synthetic wheel member")
    importing_api = put(tmp_path / "api.py",
                        b"instantiator.instantiate_non_scriptable_remote_module_template()\n")
    fields = ", ".join(f"{key}={value!r}" for key, value in probe.FORMAT_FIELDS.items())
    generator = put(tmp_path / "generator.py", (
        "_FILE_PREFIX = '_remote_module_'\n"
        "def instantiate_non_scriptable_remote_module_template():\n"
        "    generated_module_name = '_remote_module_non_scriptable'\n"
        f"    str_dict = dict({fields})\n"
        "    return _do_instantiate_remote_module_template(generated_module_name, str_dict, True)\n"
    ).encode())
    template = put(tmp_path / "template.py", (
        "_TEMPLATE_PREFIX = '# synthetic generated source\\n'\n"
        "_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA = 'pass # {args} {kwargs}\\n'\n"
    ).encode())
    generated_data = b"# synthetic generated source\npass # *args **kwargs\n"
    generated = put(tmp_path / "ephemeral" / (probe.GENERATED + ".py"), generated_data)
    generated_module = module(probe.GENERATED, generated)
    stable = dict(generator=entry(generator), template=entry(template), importing_api=entry(importing_api))
    driver_path = str(tmp_path / "absent_system" / "libcuda.so.595.58.03")
    driver = {**probe.DRIVER_BINDING, "path": driver_path}
    monkeypatch.setattr(probe, "DRIVER", driver_path)
    monkeypatch.setattr(probe, "DRIVER_BINDING", driver)
    monkeypatch.setattr(probe, "WORKER_SHA", entry(worker)["sha256"])
    monkeypatch.setattr(probe, "GENERATED_EXPECTED",
                        dict(size=len(generated_data), sha256=sha(generated_data)))
    monkeypatch.setattr(probe, "STABLE_PINS", {k: v["sha256"] for k, v in stable.items()})
    payloads = [str(tmp_path / "never-open" / f"feature{i}.pkl") for i in range(2)]
    payloads += [str(tmp_path / "never-open" / "head.pkl")]
    monkeypatch.setattr(probe, "FORBIDDEN_COUNT", len(payloads))
    baseline = dict(schema_version=1, task="hoptimus_mosaic_predictions", code_files=first,
                    features=[{"path": p} for p in payloads[:2]],
                    heads=[{"assets": {"model": {"path": payloads[2]}}}])
    baseline_path = json_file(tmp_path / "baseline_spec.json", baseline)
    declaration = dict(
        kind="deterministic_import_time_generated_source", persistent_path_required=False,
        factory="instantiate_non_scriptable_remote_module_template",
        fixed_inputs=dict(generated_module_name=probe.GENERATED,
                          enable_moving_cpu_tensors_to_cuda=True, format_fields=probe.FORMAT_FIELDS),
        template_terms_concatenated=list(probe.TERMS),
        expected_source=probe.GENERATED_EXPECTED, stable_sources=stable,
        historical_observation=dict(path=str(tmp_path / "expired" / generated.name),
                                    **probe.GENERATED_EXPECTED),
    )
    software = {e["path"]: e for e in (*first.values(), *stable.values(), entry(wheel), entry(worker))}
    software[driver_path] = driver
    profile = dict(schema_version=1, first_party_code=first, software_bindings=software,
                   generated_module_bindings={probe.GENERATED: declaration})
    profile_path = json_file(tmp_path / "baseline_profile.json", profile)
    guard_entries = {path: e["sha256"] for path, e in software.items()}
    guard_entries[str(baseline_path)] = entry(baseline_path)["sha256"]
    guard_entries[str(profile_path)] = entry(profile_path)["sha256"]
    guard_data = "".join(f"{value}  {path}\n" for path, value in sorted(guard_entries.items())).encode()
    guard_path = put(tmp_path / "baseline_guard.sha256", guard_data)
    monkeypatch.setattr(probe, "BASELINE_COUNT", len(guard_entries))
    failure = dict(schema_version=1, failed_guard=driver, worker_host=probe.HOST,
                   guard_entries_passed=len(guard_entries) - 1, worker_entry_observed=False)
    failure_path = json_file(tmp_path / "failure.json", failure)
    review = dict(schema_version=1, disposition="accepted",
                  review_kind="exact_amended_mosaic_runtime_resource_declaration",
                  spec_sha256=entry(baseline_path)["sha256"], runtime_profile_sha256=entry(profile_path)["sha256"],
                  launch_sha256=entry(guard_path)["sha256"])
    review_path = json_file(tmp_path / "runtime_review.json", review)
    sources = dict(baseline_spec=entry(baseline_path), baseline_profile=entry(profile_path),
                   baseline_guard=entry(guard_path), failure_report=entry(failure_path),
                   runtime_review=entry(review_path))
    monkeypatch.setattr(probe, "PINS", {role: e["sha256"] for role, e in sources.items()})
    spec = dict(schema_version=1, run_id="synthetic_cpu_startup", task=probe.TASK,
                expected_host=probe.HOST, output_root=str(root), sources=sources,
                code_files=dict(probe=entry(SCRIPT), worker=entry(worker)))
    spec_path = json_file(tmp_path / "spec.json", spec)
    for name in probe.THREADS:
        monkeypatch.setenv(name, "1")
    for name, value in (("CUDA_VISIBLE_DEVICES", ""), ("PYTHONDONTWRITEBYTECODE", "1"),
                        ("PYTHONHASHSEED", "0"), ("PROSPER_OUT", str(root)),
                        ("PYTHONPATH", str(package_root))):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("PROSPER_PARAMS", raising=False)
    monkeypatch.delenv("PROSPER_ROOT", raising=False)
    monkeypatch.delenv("PROSPER_MODULE", raising=False)
    monkeypatch.setattr(probe.socket, "gethostname", lambda: probe.HOST)
    monkeypatch.setattr(probe.os, "getpid", lambda: 101)
    monkeypatch.setattr(sys, "path", [str(package_root), *sys.path])
    runtime = dict(distribution_versions={"numpy": "2.2.4", "scipy": "1.16.0", "scikit-learn": "1.7.0"},
                   implementation={
                       name: dict(version="synthetic", path=first["fit"]["path"])
                       for name in ("sklearn", "numpy", "scipy")},
                   native_pools=[dict(num_threads=1, architecture="synthetic_dispatch", filepath=str(wheel))])
    runtime["implementation"]["SGDClassifier"] = dict(path=first["fit"]["path"])
    calls = []

    def deferred():
        calls.append("reader")
        return modules[probe.DIRECT["head_reader"]]

    def code_paths(reader):
        assert reader is modules[probe.DIRECT["head_reader"]]
        calls.append("code_paths")
        return {role: Path(e["path"]) for role, e in first.items()}

    def getter():
        calls.append("runtime_profile")
        return runtime

    modules[probe.MOSAIC]._reader = deferred
    modules[probe.MOSAIC].code_paths = code_paths
    modules[probe.DIRECT["fit"]]._runtime_profile = getter
    for obj in modules.values():
        for name in ("run", "fit", "predict", "predict_head", "load_saved_annotation_head",
                     "enumerate_backends", "is_available", "get_default_device"):
            setattr(obj, name, forbidden)

    original_import = probe.importlib.import_module

    def import_fixed(name, package=None):
        if name != probe.MOSAIC:
            assert not (name == "prosper" or name.startswith("prosper.") or name == "torch"
                        or name.startswith("torch.")), "actual package/native import forbidden"
            return original_import(name, package)
        assert name == probe.MOSAIC
        calls.append("import")
        for key, obj in modules.items():
            monkeypatch.setitem(sys.modules, key, obj)
        monkeypatch.setitem(sys.modules, probe.GENERATED, generated_module)
        return modules[name]

    monkeypatch.setattr(probe.importlib, "import_module", import_fixed)
    original_records = probe._module_records

    def synthetic_records(bootstrap):
        visible = {k: v for k, v in sys.modules.items()
                   if k == probe.GENERATED or k == "prosper" or k.startswith("prosper.")}
        records = {}
        for name, obj in visible.items():
            s = obj.__spec__
            records[name] = dict(name=obj.__name__, file=obj.__file__, origin=s.origin,
                                 search_locations=None if s.submodule_search_locations is None
                                 else list(s.submodule_search_locations),
                                 phase="bootstrap" if name in bootstrap else "startup")
        records["__main__"] = dict(name="__main__", file=str(SCRIPT), origin=None,
                                  search_locations=None, phase="bootstrap")
        records["synthetic_builtin"] = dict(name="synthetic_builtin", file=None, origin="built-in",
                                            search_locations=None, phase="startup")
        return records

    monkeypatch.setattr(probe, "_module_records", synthetic_records)
    monkeypatch.setattr(probe, "_mapped_paths", lambda: [str(wheel)])
    return SimpleNamespace(
        root=root, out=root / "attempt", spec=spec, spec_path=spec_path, profile=profile,
        baseline=baseline, modules=modules, generated=generated, generated_data=generated_data,
        runtime=runtime, calls=calls, payloads=payloads, guard_path=guard_path, worker=worker,
        wheel=wheel, driver=Path(driver_path), monkeypatch=monkeypatch, original_records=original_records,
    )


def run_capture(packet):
    return probe.capture(packet.spec_path, entry(packet.spec_path)["sha256"], packet.out)


def fresh(packet):
    for name in tuple(sys.modules):
        if name == probe.GENERATED or name == "prosper" or name.startswith("prosper."):
            packet.monkeypatch.delitem(sys.modules, name)
    if packet.generated.exists():
        packet.generated.unlink()
    packet.monkeypatch.setattr(probe.os, "getpid", lambda: 202)


def captured(packet):
    retained = run_capture(packet)
    handoff = put(packet.root / "launcher_handoff.json", retained)
    fresh(packet)
    return retained, handoff


def run_verify(packet, handoff, retained=None):
    data = handoff.read_bytes() if retained is None else retained
    return probe.verify(packet.spec_path, entry(packet.spec_path)["sha256"],
                        packet.out, handoff, sha(data))


@pytest.fixture
def namespace_packet(packet, monkeypatch):
    sources = {}
    defining = {}
    for name, filename in (("torch", "__init__.py"), ("torch._ops", "_ops.py"),
                           ("torch._classes", "_classes.py")):
        path = put(packet.root / "synthetic_torch" / filename,
                   f"# defining source for {name}, never executed\n".encode())
        sources[name] = entry(path)
        defining[name] = module(name, path, name == "torch")
    monkeypatch.setattr(probe, "TORCH_NAMESPACE_SOURCES", sources, raising=False)
    namespaces = {}
    for name, marker in (("torch.ops", "_ops.py"), ("torch.classes", "_classes.py")):
        namespace_type = type("SyntheticNamespace", (ModuleType,), {"__file__": marker})
        namespaces[name] = namespace_type(name)
    main = module("__main__", SCRIPT)
    main.__spec__ = None
    builtin = ModuleType("synthetic_builtin")
    builtin.__spec__ = importlib.machinery.ModuleSpec("synthetic_builtin", loader=None, origin="built-in")
    initial = {"__main__": main, "synthetic_builtin": builtin}
    proxy = SimpleNamespace(modules=dict(initial), stdout=sys.stdout,
                            executable=sys.executable, version=sys.version)
    monkeypatch.setattr(probe, "sys", proxy)
    monkeypatch.setattr(probe, "_module_records", packet.original_records)
    original_import = probe.importlib.import_module

    def observed(name, package=None):
        result = original_import(name, package)
        if name == probe.MOSAIC:
            proxy.modules.update(packet.modules)
            proxy.modules.update(defining)
            proxy.modules.update(namespaces)
            proxy.modules[probe.GENERATED] = sys.modules[probe.GENERATED]
        return result

    monkeypatch.setattr(probe.importlib, "import_module", observed)
    packet.namespace_sources = sources
    packet.namespace_modules = namespaces
    packet.namespace_defining = defining
    packet.namespace_proxy = proxy
    packet.namespace_initial = initial
    return packet


def namespace_fresh(packet):
    fresh(packet)
    packet.namespace_proxy.modules = dict(packet.namespace_initial)


@pytest.mark.parametrize("name,marker", [("torch.ops", "_ops.py"), ("torch.classes", "_classes.py")])
def test_repair_i1_actual_collector_accepts_exact_namespace_marker(monkeypatch, name, marker):
    namespace_type = type("SyntheticNamespace", (ModuleType,), {"__file__": marker})
    monkeypatch.setattr(probe, "sys", SimpleNamespace(modules={name: namespace_type(name)}))
    records = probe._module_records([])
    assert records[name] == dict(name=name, file=marker, origin=None,
                                 search_locations=None, phase="startup")
    probe.module_schema(records)


def test_repair_i1_actual_collector_full_capture_and_verify(namespace_packet):
    packet = namespace_packet
    retained = run_capture(packet)
    handoff = put(packet.root / "namespace_handoff.json", retained)
    profile = probe.strict_json((packet.out / probe.CAPTURE_NAMES[0]).read_bytes())
    for name, marker in (("torch.ops", "_ops.py"), ("torch.classes", "_classes.py")):
        assert profile["modules"][name]["file"] == marker
        assert profile["modules"][name]["origin"] is None
        provenance = profile["non_file_namespaces"][name]
        assert provenance["reported_file"] == marker
        assert set(provenance["sources"]) == {"torch", provenance["defining_module"]}
        for source in provenance["sources"].values():
            assert profile["software_bindings"][source["path"]] == source
    namespace_fresh(packet)
    result = run_verify(packet, handoff, retained)
    assert (packet.out / probe.FINAL_NAME).exists()
    for source in packet.namespace_sources.values():
        assert source["path"] in result["checked_bindings"]
    assert "torch" not in sys.modules


@pytest.mark.parametrize("case", ["wrong-marker", "absolute-marker", "alias-name", "origin", "search",
                                 "unknown-relative", "unknown-missing"])
def test_repair_i1_only_exact_non_file_namespaces_are_exempt(namespace_packet, monkeypatch, case):
    packet = namespace_packet
    original = probe._runtime

    def changed():
        result = original()
        namespace = packet.namespace_modules["torch.ops"]
        if case == "wrong-marker":
            namespace.__file__ = "_wrong.py"
        elif case == "absolute-marker":
            namespace.__file__ = packet.namespace_sources["torch._ops"]["path"]
        elif case == "alias-name":
            namespace.__name__ = "torch.alias"
        elif case == "origin":
            namespace.__spec__ = importlib.machinery.ModuleSpec("torch.ops", None, origin="_ops.py")
        elif case == "search":
            namespace.__spec__ = importlib.machinery.ModuleSpec("torch.ops", None, is_package=True)
            namespace.__spec__.submodule_search_locations = [str(packet.root)]
        else:
            obj = ModuleType("unknown_namespace")
            obj.__file__ = "_ops.py" if case == "unknown-relative" else str(packet.root / "missing.py")
            packet.namespace_proxy.modules[obj.__name__] = obj
        return result

    monkeypatch.setattr(probe, "_runtime", changed)
    with pytest.raises((probe.ProbeError, FileNotFoundError)):
        run_capture(packet)
    assert not (packet.out / probe.FINAL_NAME).exists()


@pytest.mark.parametrize("case", ["missing-definition", "wrong-definition-origin", "changed-source",
                                 "changed-initializer"])
def test_repair_i1_defining_source_pins_are_checked(namespace_packet, monkeypatch, case):
    packet = namespace_packet
    original = probe._runtime

    def changed():
        result = original()
        if case == "missing-definition":
            del packet.namespace_proxy.modules["torch._ops"]
        elif case == "wrong-definition-origin":
            packet.namespace_defining["torch._ops"].__spec__.origin = str(packet.worker)
        else:
            name = "torch._ops" if case == "changed-source" else "torch"
            path = Path(packet.namespace_sources[name]["path"])
            data = bytearray(path.read_bytes())
            data[-2] ^= 1
            path.write_bytes(data)
        return result

    monkeypatch.setattr(probe, "_runtime", changed)
    with pytest.raises(probe.ProbeError, match="defining|digest"):
        run_capture(packet)


@pytest.mark.parametrize("case", ["namespace-omitted", "source-unbound", "marker-forged"])
def test_repair_i1_namespace_provenance_is_reconciled_in_verify(namespace_packet, case):
    packet = namespace_packet
    retained = run_capture(packet)
    profile = probe.strict_json((packet.out / probe.CAPTURE_NAMES[0]).read_bytes())
    namespace_fresh(packet)
    if case == "namespace-omitted":
        del profile["non_file_namespaces"]["torch.ops"]
    elif case == "source-unbound":
        profile["non_file_namespaces"]["torch.ops"]["sources"]["torch._ops"]["sha256"] = "a" * 64
    else:
        profile["non_file_namespaces"]["torch.ops"]["reported_file"] = "_wrong.py"
    context = probe.prepare(packet.spec_path, entry(packet.spec_path)["sha256"], packet.out)
    with pytest.raises(probe.ProbeError, match="namespace provenance"):
        probe.validate_profile(profile, context, probe.strict_json(retained), probe.Inventory(packet.payloads))


@pytest.mark.parametrize("kind", ["different-content", "same-content"])
def test_repair_i2_original_software_alias_retarget_rejected(tmp_path, kind):
    target = put(tmp_path / "target.py", b"abc")
    other = put(tmp_path / "other.py", b"abc" if kind == "same-content" else b"xyz")
    alias = tmp_path / "alias.py"
    alias.symlink_to(target)
    inventory = probe.Inventory()
    inventory.read(alias)
    inventory.sweep()
    alias.unlink()
    alias.symlink_to(other)
    with pytest.raises(probe.ProbeError, match="declared pathname"):
        inventory.sweep()


def test_repair_i2_late_handoff_alias_retarget_cleans_receipt(packet, monkeypatch):
    retained, handoff = captured(packet)
    alias = packet.root / "handoff_alias.json"
    alias.symlink_to(handoff)
    foreign = put(packet.root / "foreign_handoff.json", b"not the retained handoff\n")
    original = probe.OwnedDirectory.write

    def retarget(self, name, data):
        result = original(self, name, data)
        if name == probe.FINAL_NAME:
            alias.unlink()
            alias.symlink_to(foreign)
        return result

    monkeypatch.setattr(probe.OwnedDirectory, "write", retarget)
    with pytest.raises(probe.ProbeError, match="declared pathname"):
        run_verify(packet, alias, retained)
    assert handoff.read_bytes() == retained and alias.resolve() == foreign
    assert not (packet.out / probe.FINAL_NAME).exists()


def test_repair_i2_unchanged_multiple_aliases_retain_names_and_limits(tmp_path):
    target = put(tmp_path / "target.py", b"abc")
    one, two = tmp_path / "one.py", tmp_path / "two.py"
    one.symlink_to(target)
    two.symlink_to(one)
    inventory = probe.Inventory()
    _, first = inventory.read(one, limit=3)
    _, second = inventory.read(two, limit=4)
    inventory.sweep()
    assert first == second
    assert set(inventory.consumed_paths) == {str(one), str(two)}
    assert inventory.path_limits == {str(one): 3, str(two): 4}
    assert set(inventory.records) == {str(target)}
    assert inventory.total == 3


@pytest.mark.parametrize("name", ["persistent", "handoff"])
def test_repair_i2_unchanged_alias_capture_verify_provenance(packet, monkeypatch, name):
    retained, handoff = captured(packet)
    alias = packet.root / "unchanged_alias.json"
    alias.symlink_to(handoff)
    if name == "persistent":
        # A fresh synthetic capture includes a file-backed non-first-party alias.
        packet.out.rename(packet.root / "retained_first_capture")
        packet.generated.parent.mkdir(exist_ok=True)
        put(packet.generated, packet.generated_data)
        alias.unlink()
        alias.symlink_to(packet.wheel)
        original = probe._module_records

        def record_alias(bootstrap):
            records = original(bootstrap)
            records["synthetic_alias"] = dict(name="synthetic_alias", file=str(alias),
                                             origin=str(alias), search_locations=None, phase="startup")
            return records

        monkeypatch.setattr(probe, "_module_records", record_alias)
        monkeypatch.setattr(probe.os, "getpid", lambda: 101)
        retained = run_capture(packet)
        handoff = put(packet.root / "second_handoff.json", retained)
        fresh(packet)
        profile = probe.strict_json((packet.out / probe.CAPTURE_NAMES[0]).read_bytes())
        assert profile["consumed_paths"][str(alias)]["path"] == str(packet.wheel)
        run_verify(packet, handoff, retained)
    else:
        result = run_verify(packet, alias, retained)
        assert result["capture_handoff"]["path"] == str(handoff)
        assert result["capture_handoff_path"] == dict(path=str(alias), target=result["capture_handoff"])


def test_repair_i2_capture_alias_retarget_between_processes_rejected(packet, monkeypatch):
    alias = packet.root / "declared_module_alias.py"
    alias.symlink_to(packet.wheel)
    original = probe._module_records

    def record_alias(bootstrap):
        records = original(bootstrap)
        records["synthetic_alias"] = dict(name="synthetic_alias", file=str(alias),
                                         origin=str(alias), search_locations=None, phase="startup")
        return records

    monkeypatch.setattr(probe, "_module_records", record_alias)
    retained, handoff = captured(packet)
    replacement = put(packet.root / "byte_identical_foreign.py", packet.wheel.read_bytes())
    alias.unlink()
    alias.symlink_to(replacement)
    with pytest.raises(probe.ProbeError, match="consumed pathname"):
        run_verify(packet, handoff, retained)
    assert not (packet.out / probe.FINAL_NAME).exists()


@pytest.mark.parametrize("case", ["missing", "extra", "boolean-size", "target-mismatch"])
def test_repair_i2_captured_original_path_schema_and_coverage(packet, case):
    retained, handoff = captured(packet)
    profile = probe.strict_json((packet.out / probe.CAPTURE_NAMES[0]).read_bytes())
    path = next(iter(profile["consumed_paths"]))
    if case == "missing":
        del profile["consumed_paths"][path]
    elif case == "extra":
        profile["consumed_paths"][path]["extra"] = True
    elif case == "boolean-size":
        profile["consumed_paths"][path]["size"] = True
    else:
        profile["consumed_paths"][path]["path"] = str(handoff)
    context = probe.prepare(packet.spec_path, entry(packet.spec_path)["sha256"], packet.out)
    with pytest.raises(probe.ProbeError, match="consumed pathname"):
        probe.validate_profile(profile, context, probe.strict_json(retained), probe.Inventory(packet.payloads))


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("reuse", ["natural", "dup2"])
@pytest.mark.parametrize("replace_directory", [False, True])
def test_repair_i3_foreign_fd_reuse_preserves_fd_and_exact_abort(
        packet, monkeypatch, kind, reuse, replace_directory):
    retained, handoff = captured(packet)
    original_identity = (packet.out.stat().st_dev, packet.out.stat().st_ino)
    foreign = packet.root / "foreign_descriptor"
    if kind == "file":
        put(foreign, b"unrelated file, retain contents and offset\n")
    else:
        foreign.mkdir()
        put(foreign / "retained", b"foreign directory")
    original_close, original_read = probe.close, probe.os.read
    primary = KeyboardInterrupt("exact reproduced post-close descriptor-reuse cancellation")
    foreign_fds, foreign_reads, foreign_closes = [], [], []
    triggered = False

    def inspected_read(fd, size):
        if fd in foreign_fds:
            foreign_reads.append(fd)
        return original_read(fd, size)

    def close_then_reuse(fd, error=None):
        nonlocal triggered
        if fd in foreign_fds:
            foreign_closes.append(fd)
        observed = os.fstat(fd)
        eligible = stat.S_ISDIR(observed.st_mode) and \
            (observed.st_dev, observed.st_ino) == original_identity and error is None and \
            (packet.out / probe.FINAL_NAME).exists() and not triggered
        original_close(fd, error)
        if eligible:
            triggered = True
            if reuse == "natural":
                replacement_fd = os.open(foreign, os.O_RDONLY)
                assert replacement_fd == fd
            else:
                placeholder = os.open(foreign, os.O_RDONLY)
                assert placeholder == fd
                other = os.open(foreign, os.O_RDONLY)
                assert other != fd
                os.dup2(other, fd)
                os.close(other)
            foreign_fds.append(fd)
            if replace_directory:
                replacement = packet.root / "foreign_output_after_close"
                shutil.copytree(packet.out, replacement)
                packet.out.rename(packet.root / "preserved_original_output")
                replacement.rename(packet.out)
            raise primary

    monkeypatch.setattr(probe, "close", close_then_reuse)
    monkeypatch.setattr(probe.os, "read", inspected_read)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            run_verify(packet, handoff, retained)
        assert caught.value is primary and triggered
        assert not foreign_reads and not foreign_closes
        assert (packet.out / probe.FINAL_NAME).exists() is replace_directory
        for fd in foreign_fds:
            assert (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == \
                (foreign.stat().st_dev, foreign.stat().st_ino)
            assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        if kind == "file":
            assert foreign.read_bytes() == b"unrelated file, retain contents and offset\n"
        else:
            assert (foreign / "retained").read_bytes() == b"foreign directory"
    finally:
        for fd in foreign_fds:
            os.close(fd)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_repair_i3_cleanup_then_final_release_never_closes_foreign_fd(tmp_path, kind):
    out = tmp_path / "out"
    owned = probe.OwnedDirectory(out, create=True)
    owned.write(probe.FINAL_NAME, b"synthetic verification receipt\n")
    foreign = tmp_path / "unrelated"
    if kind == "file":
        put(foreign, b"retained")
    else:
        foreign.mkdir()
    descriptor = owned.fd
    os.close(descriptor)
    replacement = os.open(foreign, os.O_RDONLY)
    assert replacement == descriptor
    primary = KeyboardInterrupt("cleanup-before-finally descriptor reuse")
    try:
        owned.remove_owned(probe.FINAL_NAME, primary)
        owned.release(primary)
        assert not (out / probe.FINAL_NAME).exists()
        assert (os.fstat(replacement).st_dev, os.fstat(replacement).st_ino) == \
            (foreign.stat().st_dev, foreign.stat().st_ino)
        assert os.lseek(replacement, 0, os.SEEK_CUR) == 0
    finally:
        os.close(replacement)


def test_top_level_stdlib_only_no_package_registration():
    tree = ast.parse(SCRIPT.read_bytes())
    imported = []
    for n in tree.body:
        if isinstance(n, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in n.names)
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module.split(".")[0])
    assert set(imported) <= sys.stdlib_module_names | {"__future__"}
    assert "prosper" not in sys.modules and "torch" not in sys.modules


def test_capture_and_fresh_verify_complete_synthetic_route(packet):
    retained, handoff = captured(packet)
    assert len(retained) <= probe.HANDOFF_LIMIT and retained.count(b"\n") == 1
    receipt = probe.strict_json(retained)
    probe.receipt_schema(receipt)
    assert packet.calls == ["import", "reader", "code_paths", "runtime_profile"]
    result = run_verify(packet, handoff, retained)
    assert result["verify_pid"] != result["capture_pid"]
    assert set(p.name for p in packet.out.iterdir()) == {*probe.CAPTURE_NAMES, probe.FINAL_NAME}
    assert str(packet.wheel) in result["checked_bindings"]  # GPU name is NOT omitted.
    assert str(packet.generated) not in result["checked_bindings"]
    assert "prosper" not in sys.modules and "torch" not in sys.modules


@pytest.mark.parametrize("version", [True, 1.0, 0, 2, "1"])
def test_strict_spec_version_before_package(packet, version):
    packet.spec["schema_version"] = version
    json_file(packet.spec_path, packet.spec)
    with pytest.raises(probe.ProbeError, match="schema_version"):
        run_capture(packet)
    assert packet.calls == []


@pytest.mark.parametrize("field", ["extra", "missing", "task", "host", "relative", "bad-pin", "worker"])
def test_exact_spec_schema_and_pins(packet, field):
    if field == "extra":
        packet.spec["extra"] = True
    elif field == "missing":
        del packet.spec["code_files"]
    elif field == "task":
        packet.spec["task"] = "scoring"
    elif field == "host":
        packet.spec["expected_host"] = "other"
    elif field == "relative":
        packet.spec["output_root"] = "relative"
    elif field == "bad-pin":
        packet.spec["sources"]["failure_report"]["sha256"] = "a" * 64
    else:
        packet.spec["code_files"]["worker"]["sha256"] = "a" * 64
    json_file(packet.spec_path, packet.spec)
    with pytest.raises((probe.ProbeError, KeyError)):
        run_capture(packet)
    assert packet.calls == []


@pytest.mark.parametrize("bad", [
    b'{"schema_version":1,"schema_version":1}', b'{"schema_version":1,"x":NaN}',
    b'{"schema_version":1,"x":1e999}', b"\xff", b"{",
])
def test_strict_json(bad):
    with pytest.raises(probe.ProbeError):
        probe.strict_json(bad)


@pytest.mark.parametrize("encoding", ["utf-16-be", "utf-16-le", "utf-32-be", "utf-32-le"])
def test_json_controls_and_handoff_require_utf8(encoding):
    with pytest.raises(probe.ProbeError):
        probe.strict_json('{"schema_version":1}\n'.encode(encoding))


@pytest.mark.parametrize("name,value", [("OMP_NUM_THREADS", "2"), ("CUDA_VISIBLE_DEVICES", "0"),
                                      ("PYTHONDONTWRITEBYTECODE", "0"), ("PYTHONHASHSEED", "1"),
                                      ("PROSPER_PARAMS", ""), ("PROSPER_OUT", "/missing"),
                                      ("PYTHONPATH", "/wrong:other")])
def test_environment_failure_precedes_package(packet, name, value):
    packet.monkeypatch.setenv(name, value)
    with pytest.raises((probe.ProbeError, FileNotFoundError)):
        run_capture(packet)
    assert packet.calls == []


def test_wrong_actual_host_before_import(packet):
    packet.monkeypatch.setattr(probe.socket, "gethostname", lambda: "wrong_host")
    with pytest.raises(probe.ProbeError, match="host"):
        run_capture(packet)
    assert packet.calls == []


@pytest.mark.parametrize("context", ["capture_direct_scripts", "worker_help_cwd"])
def test_shadow_root_rejected_before_initializer_without_path_repair(packet, context):
    shadow = packet.root / ("scripts" if context.startswith("capture") else "worker_cwd")
    sentinel = packet.root / "INITIALIZER_EXECUTED"
    put(shadow / "prosper" / "__init__.py", f"open({str(sentinel)!r}, 'w').write('bad')\n".encode())
    packet.monkeypatch.setattr(sys, "path", [str(shadow), *sys.path])
    before = list(sys.path)
    with pytest.raises(probe.ProbeError, match="origin mismatch"):
        probe.check_effective_prosper_root(packet.baseline["code_files"]["package_init"]["path"])
    assert not sentinel.exists() and sys.path == before


def test_capture_checks_its_effective_script_precedence_before_import(packet):
    scripts = packet.root / "direct_scripts"
    sentinel = packet.root / "shadow_ran"
    put(scripts / "prosper" / "__init__.py", f"open({str(sentinel)!r},'w').write('bad')\n".encode())
    packet.monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    original = list(sys.path)
    with pytest.raises(probe.ProbeError, match="before initializer"):
        run_capture(packet)
    assert packet.calls == [] and not sentinel.exists() and sys.path == original


@pytest.mark.parametrize("name", ["prosper", "prosper.cached"])
def test_cached_package_rejected_not_repaired(packet, name):
    cache = ModuleType(name)
    packet.monkeypatch.setitem(sys.modules, name, cache)
    with pytest.raises(probe.ProbeError, match="cached"):
        run_capture(packet)
    assert sys.modules[name] is cache and packet.calls == []


@pytest.mark.parametrize("case", ["unexpected", "missing", "alias", "origin", "search"])
def test_actual_inventory_independently_reconciled(packet, case):
    original = probe._runtime

    def changed():
        paths, runtime = original()
        name = probe.DIRECT["head_reader"]
        if case == "unexpected":
            packet.monkeypatch.setitem(sys.modules, "prosper.extra", module("prosper.extra", packet.worker))
        elif case == "missing":
            packet.monkeypatch.delitem(sys.modules, name)
        elif case == "alias":
            packet.modules[name].__name__ = "prosper.alias"
        elif case == "origin":
            packet.modules[name].__spec__.origin = str(packet.worker)
        else:
            packet.modules["prosper"].__spec__.submodule_search_locations = [str(packet.root)]
        return paths, runtime

    packet.monkeypatch.setattr(probe, "_runtime", changed)
    with pytest.raises(probe.ProbeError, match="inventory|origin|search"):
        run_capture(packet)


@pytest.mark.parametrize("suffix", [".pkl", ".npy", ".pt", ".pth", ".safetensors"])
def test_known_payload_suffix_never_opened(tmp_path, monkeypatch, suffix):
    payload = put(tmp_path / ("payload" + suffix), b"synthetic forbidden")
    monkeypatch.setattr(probe.os, "open", forbidden)
    with pytest.raises(probe.ProbeError, match="payload"):
        probe.Inventory().read(payload)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_payload_alias_never_opened(tmp_path, monkeypatch, kind):
    payload = put(tmp_path / "payload.pkl", b"synthetic forbidden")
    alias = tmp_path / "software.py"
    if kind == "symlink":
        alias.symlink_to(payload)
    else:
        os.link(payload, alias)
    inventory = probe.Inventory([str(payload)])
    monkeypatch.setattr(probe.os, "open", forbidden)
    with pytest.raises(probe.ProbeError, match="payload"):
        inventory.read(alias)


@pytest.mark.parametrize("kind", ["missing-other", "changed-other", "driver-present", "old-temp-in-guard"])
def test_only_exact_recorded_driver_can_be_omitted(packet, kind):
    if kind == "missing-other":
        packet.wheel.unlink()
    elif kind == "changed-other":
        packet.wheel.write_bytes(b"changed")
    elif kind == "driver-present":
        put(packet.driver, b"changed precondition")
    else:
        old = packet.profile["generated_module_bindings"][probe.GENERATED]["historical_observation"]["path"]
        data = packet.guard_path.read_bytes() + f"{'a'*64}  {old}\n".encode()
        put(packet.guard_path, data)
        packet.spec["sources"]["baseline_guard"] = entry(packet.guard_path)
        packet.monkeypatch.setitem(probe.PINS, "baseline_guard", sha(data))
        # The pinned runtime-review link still rejects this coherent-question change.
        json_file(packet.spec_path, packet.spec)
    with pytest.raises((probe.ProbeError, FileNotFoundError)):
        run_capture(packet)
    assert packet.calls == []


@pytest.mark.parametrize("case", ["bytes", "origin", "missing", "unknown-ephemeral"])
def test_generated_source_lifetime_and_unknown_ephemeral(packet, case):
    original = probe._runtime

    def changed():
        result = original()
        if case == "bytes":
            packet.generated.write_bytes(b"changed")
        elif case == "origin":
            sys.modules[probe.GENERATED].__spec__.origin = str(packet.worker)
        elif case == "missing":
            packet.generated.unlink()
        else:
            packet.monkeypatch.setitem(sys.modules, "prosper.extra",
                                      module("prosper.extra", packet.root / "deleted.py"))
        return result

    packet.monkeypatch.setattr(probe, "_runtime", changed)
    with pytest.raises((probe.ProbeError, FileNotFoundError)):
        run_capture(packet)


def test_driver_absence_cannot_change_during_capture(packet):
    original = probe._runtime

    def changed():
        result = original()
        put(packet.driver, b"precondition changed during startup")
        return result

    packet.monkeypatch.setattr(probe, "_runtime", changed)
    with pytest.raises(probe.ProbeError, match="no longer absent"):
        run_capture(packet)


@pytest.mark.parametrize("field", ["generator", "template", "importing_api"])
def test_pinned_generated_derivative_sources_changed_before_execution(packet, field):
    path = Path(packet.profile["generated_module_bindings"][probe.GENERATED]["stable_sources"][field]["path"])
    path.write_bytes(path.read_bytes() + b"#mutation\n")
    with pytest.raises(probe.ProbeError, match="size|digest|content"):
        run_capture(packet)
    assert packet.calls == []


def test_no_generic_missing_ephemeral_exclusion(packet):
    original_records = probe._module_records

    def changed(bootstrap):
        records = original_records(bootstrap)
        records["unknown_ephemeral_module"] = dict(
            name="unknown_ephemeral_module", file=str(packet.root / "not_present.py"),
            origin=str(packet.root / "not_present.py"), search_locations=None, phase="startup")
        return records

    packet.monkeypatch.setattr(probe, "_module_records", changed)
    with pytest.raises(FileNotFoundError):
        run_capture(packet)


def test_non_unit_pools_fail_no_configuration_repair(packet):
    packet.runtime["native_pools"][0]["num_threads"] = 2
    with pytest.raises(probe.ProbeError, match="non-unit"):
        run_capture(packet)
    assert packet.runtime["native_pools"][0]["num_threads"] == 2


def test_startup_stdout_is_not_filtered(packet):
    original = probe._runtime

    def noisy():
        print("stray stdout")
        return original()

    packet.monkeypatch.setattr(probe, "_runtime", noisy)
    with pytest.raises(probe.ProbeError, match="stdout"):
        run_capture(packet)
    assert packet.out.is_dir()


def test_inventory_running_growth_without_large_allocation(tmp_path, monkeypatch):
    path = put(tmp_path / "source.py", b"abc")
    original = probe.os.read
    changed = False

    def growth(fd, size):
        nonlocal changed
        data = original(fd, size)
        if not changed:
            changed = True
            path.write_bytes(b"abcdef")
        return data

    monkeypatch.setattr(probe.os, "read", growth)
    with pytest.raises(probe.ProbeError, match="running"):
        probe.Inventory().read(path, limit=4)


@pytest.mark.parametrize("kind", ["same-content-replacement", "truncate", "resolution"])
def test_in_read_identity_size_resolution_checks(tmp_path, monkeypatch, kind):
    real = put(tmp_path / "source.py", b"abc")
    path = tmp_path / "alias.py"
    path.symlink_to(real)
    original = probe.os.read
    changed = False

    def mutation(fd, count):
        nonlocal changed
        data = original(fd, count)
        if not changed:
            changed = True
            if kind == "same-content-replacement":
                real.rename(tmp_path / "kept_source.py")
                put(real, b"abc")
            elif kind == "truncate":
                real.write_bytes(b"a")
            else:
                other = put(tmp_path / "other.py", b"abc")
                path.unlink()
                path.symlink_to(other)
        return data

    monkeypatch.setattr(probe.os, "read", mutation)
    with pytest.raises(probe.ProbeError, match="truncated|changed|replaced"):
        probe.Inventory().read(path)


@pytest.mark.parametrize("case", ["fifo", "directory", "size", "aggregate", "count"])
def test_regular_and_inventory_caps_precede_reads(tmp_path, monkeypatch, case):
    path = tmp_path / "source.py"
    if case == "fifo":
        os.mkfifo(path)
    elif case == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"abc")
    if case == "aggregate":
        monkeypatch.setattr(probe, "TOTAL_LIMIT", 2)
    elif case == "count":
        monkeypatch.setattr(probe, "FILE_COUNT", 0)
    monkeypatch.setattr(probe.os, "open", forbidden)
    with pytest.raises(probe.ProbeError):
        probe.Inventory().read(path, limit=2 if case == "size" else 10)


def test_foreign_existing_output_directory_not_adopted(packet):
    packet.out.mkdir()
    put(packet.out / "foreign", b"keep")
    with pytest.raises(FileExistsError):
        run_capture(packet)
    assert (packet.out / "foreign").read_bytes() == b"keep"


def test_capture_control_mutation_detected_on_final_sweep(packet):
    original = probe._runtime

    def changed():
        result = original()
        path = Path(packet.spec["sources"]["failure_report"]["path"])
        data = path.read_bytes()
        path.write_bytes(data.replace(b"false", b"true "))
        return result

    packet.monkeypatch.setattr(probe, "_runtime", changed)
    with pytest.raises(probe.ProbeError, match="digest|content"):
        run_capture(packet)


def test_verify_authenticates_handoff_before_consuming_artifacts(packet, monkeypatch):
    retained, handoff = captured(packet)
    monkeypatch.setattr(probe.OwnedDirectory, "read", forbidden)
    with pytest.raises(probe.ProbeError, match="handoff digest"):
        probe.verify(packet.spec_path, entry(packet.spec_path)["sha256"], packet.out, handoff, "a" * 64)


def test_handoff_must_be_external_and_exclusive_launcher_evidence(packet):
    retained, handoff = captured(packet)
    internal = put(packet.out / "unapproved_handoff.json", retained)
    with pytest.raises(probe.ProbeError, match="outside"):
        run_verify(packet, internal, retained)
    internal.unlink()
    put(packet.out / "foreign", b"foreign")
    with pytest.raises(probe.ProbeError, match="two capture"):
        run_verify(packet, handoff, retained)


def test_schema_valid_native_profile_edit_rejected_by_retained_bytes(packet):
    retained, handoff = captured(packet)
    path = packet.out / probe.CAPTURE_NAMES[0]
    profile = probe.strict_json(path.read_bytes())
    profile["runtime_profile"]["native_pools"][0]["architecture"] = "different_architecture"
    path.write_bytes(probe.encode(profile))
    with pytest.raises(probe.ProbeError, match="artifact"):
        run_verify(packet, handoff, retained)
    assert not (packet.out / probe.FINAL_NAME).exists()


@pytest.mark.parametrize("case", [
    "limitations", "generated-extra", "generated-bool", "generated-name",
    "bootstrap-bool", "module-file-bool", "module-extra", "module-phase",
    "mappings-dict", "runtime-pool-bool", "runtime-extra", "implementation-path",
    "cpu-count-bool", "affinity-duplicate", "probe-inode-float",
])
def test_profile_semantic_schema_on_authenticated_synthetic_graph(packet, case):
    retained, handoff = captured(packet)
    receipt = probe.strict_json(retained)
    profile = probe.strict_json((packet.out / probe.CAPTURE_NAMES[0]).read_bytes())
    if case == "limitations":
        profile["limitations"] = []
    elif case == "generated-extra":
        profile["generated_module"]["extra"] = True
    elif case == "generated-bool":
        profile["generated_module"]["derivation"]["fixed_inputs"]["enable_moving_cpu_tensors_to_cuda"] = 1
    elif case == "generated-name":
        profile["generated_module"]["name"] = "other"
    elif case == "bootstrap-bool":
        profile["bootstrap_modules"][0] = True
    elif case.startswith("module-"):
        record = profile["modules"]["synthetic_builtin"]
        if case == "module-file-bool":
            record["file"] = True
        elif case == "module-extra":
            record["extra"] = True
        else:
            record["phase"] = "bootstrap"
    elif case == "mappings-dict":
        profile["executable_mappings"] = {}
    elif case == "runtime-pool-bool":
        profile["runtime_profile"]["native_pools"][0]["num_threads"] = True
    elif case == "runtime-extra":
        profile["runtime_profile"]["extra"] = True
    elif case == "implementation-path":
        unbound = put(packet.root / "unbound_implementation.py", b"# synthetic, not executed\n")
        profile["runtime_profile"]["implementation"]["numpy"]["path"] = str(unbound)
    elif case == "cpu-count-bool":
        profile["cpu"]["count"] = True
    elif case == "affinity-duplicate":
        profile["cpu"]["affinity"] = [0, 0]
    else:
        profile["probe_source"]["st_ino"] = float(profile["probe_source"]["st_ino"])
    context = probe.prepare(packet.spec_path, entry(packet.spec_path)["sha256"], packet.out)
    with pytest.raises(probe.ProbeError):
        probe.validate_profile(profile, context, receipt, probe.Inventory(packet.payloads))
    assert not (packet.out / probe.FINAL_NAME).exists()


def test_bootstrap_no_file_and_dispatch_accounting_are_explicit(packet):
    retained, handoff = captured(packet)
    profile = probe.strict_json((packet.out / probe.CAPTURE_NAMES[0]).read_bytes())
    assert profile["modules"]["__main__"]["phase"] == "bootstrap"
    assert profile["modules"]["synthetic_builtin"] == dict(
        name="synthetic_builtin", file=None, origin="built-in", search_locations=None, phase="startup")
    assert profile["runtime_profile"]["native_pools"][0]["architecture"] == "synthetic_dispatch"
    assert profile["probe_source"]["path"] == str(SCRIPT.resolve())
    assert "bootstrap_modules_are_not_all_probe_only" in profile["limitations"]
    assert profile["persistent_delta"]["dropped"] == [str(packet.driver)]
    run_verify(packet, handoff, retained)


@pytest.mark.parametrize("case", ["directory-bool", "artifact-bool", "extra", "missing", "bad-digest"])
def test_exact_nested_handoff_schema(packet, case):
    retained, handoff = captured(packet)
    receipt = probe.strict_json(retained)
    if case == "directory-bool":
        receipt["output_directory"]["st_dev"] = True
    elif case == "artifact-bool":
        receipt["artifacts"][probe.CAPTURE_NAMES[0]]["size"] = True
    elif case == "extra":
        receipt["artifacts"][probe.CAPTURE_NAMES[0]]["extra"] = 1
    elif case == "missing":
        del receipt["artifacts"][probe.CAPTURE_NAMES[1]]
    else:
        receipt["artifacts"][probe.CAPTURE_NAMES[0]]["sha256"] = "A" * 64
    data = probe.encode(receipt)
    put(handoff, data)
    with pytest.raises(probe.ProbeError):
        run_verify(packet, handoff, data)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_source_read_abort_closes_descriptor_and_preserves_exception(tmp_path, monkeypatch, exception_type):
    path = put(tmp_path / "source.py", b"abc")
    original_open, original_close = probe.os.open, probe.os.close
    primary = exception_type("synthetic source cancellation")
    acquired, closed = [], []

    def opened(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        acquired.append(fd)
        return fd

    def released(fd):
        original_close(fd)
        closed.append(fd)

    def abort(*args):
        raise primary

    monkeypatch.setattr(probe.os, "open", opened)
    monkeypatch.setattr(probe.os, "close", released)
    monkeypatch.setattr(probe.os, "read", abort)
    with pytest.raises(exception_type) as caught:
        probe.Inventory().read(path)
    assert caught.value is primary and acquired == closed


@pytest.mark.parametrize("case", ["file", "directory"])
def test_same_content_foreign_replacement_rejected(packet, case):
    retained, handoff = captured(packet)
    if case == "file":
        target = packet.out / probe.CAPTURE_NAMES[0]
        contents = target.read_bytes()
        target.rename(packet.out / "kept_original.json")
        put(target, contents)
        (packet.out / "kept_original.json").rename(packet.root / "kept_original.json")
    else:
        replacement = packet.root / "replacement"
        shutil.copytree(packet.out, replacement)
        packet.out.rename(packet.root / "kept_original_directory")
        replacement.rename(packet.out)
    with pytest.raises(probe.ProbeError, match="identity"):
        run_verify(packet, handoff, retained)


@pytest.mark.parametrize("field,value", [
    ("run_id", "stale_run"), ("spec_sha256", "a" * 64), ("capture_pid", 202),
    ("capture_pid", True), ("hostname", "other"), ("schema_version", 1.0),
])
def test_wrong_or_stale_handoff(packet, field, value):
    retained, handoff = captured(packet)
    receipt = probe.strict_json(retained)
    receipt[field] = value
    data = probe.encode(receipt)
    put(handoff, data)
    with pytest.raises(probe.ProbeError):
        run_verify(packet, handoff, data)


def test_capture_temp_must_be_absent_at_fresh_verify(packet):
    retained, handoff = captured(packet)
    put(packet.generated, packet.generated_data)
    with pytest.raises(probe.ProbeError, match="absent"):
        run_verify(packet, handoff, retained)


def test_same_pid_rejected_before_profile_consumption(packet):
    retained, handoff = captured(packet)
    packet.monkeypatch.setattr(probe.os, "getpid", lambda: 101)
    with pytest.raises(probe.ProbeError, match="fresh PID"):
        run_verify(packet, handoff, retained)


def test_verify_directory_continuity_checked_during_artifact_read(packet, monkeypatch):
    retained, handoff = captured(packet)
    original = probe.OwnedDirectory.read
    changed = False

    def replace_directory(self, name, expected):
        nonlocal changed
        if not changed:
            changed = True
            replacement = packet.root / "late_directory"
            shutil.copytree(packet.out, replacement)
            packet.out.rename(packet.root / "kept_directory")
            replacement.rename(packet.out)
        return original(self, name, expected)

    monkeypatch.setattr(probe.OwnedDirectory, "read", replace_directory)
    with pytest.raises(probe.ProbeError, match="custody"):
        run_verify(packet, handoff, retained)
    assert not (packet.out / probe.FINAL_NAME).exists()


def test_verify_has_no_package_runtime_callbacks(packet, monkeypatch):
    retained, handoff = captured(packet)
    monkeypatch.setattr(probe, "_runtime", forbidden)
    monkeypatch.setattr(probe.importlib, "import_module", forbidden)
    run_verify(packet, handoff, retained)


def test_repeat_verify_and_foreign_file_do_not_overwrite(packet):
    retained, handoff = captured(packet)
    run_verify(packet, handoff, retained)
    contents = (packet.out / probe.FINAL_NAME).read_bytes()
    with pytest.raises(probe.ProbeError, match="two capture"):
        run_verify(packet, handoff, retained)
    assert (packet.out / probe.FINAL_NAME).read_bytes() == contents


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_late_verify_failure_removes_only_own_receipt_and_preserves_error(packet, exception_type, monkeypatch):
    retained, handoff = captured(packet)
    original = probe.OwnedDirectory.write
    primary = exception_type("late synthetic verification abort")

    def late(self, name, data):
        result = original(self, name, data)
        if name == probe.FINAL_NAME:
            raise primary
        return result

    monkeypatch.setattr(probe.OwnedDirectory, "write", late)
    with pytest.raises(exception_type) as caught:
        run_verify(packet, handoff, retained)
    assert caught.value is primary
    assert set(p.name for p in packet.out.iterdir()) == set(probe.CAPTURE_NAMES)


def test_late_failure_preserves_foreign_receipt_replacement(packet, monkeypatch):
    retained, handoff = captured(packet)
    original = probe.OwnedDirectory.write
    primary = KeyboardInterrupt("replacement abort")

    def replaced(self, name, data):
        result = original(self, name, data)
        if name == probe.FINAL_NAME:
            path = packet.out / name
            path.rename(packet.root / "kept_original_receipt")
            put(path, b"foreign")
            raise primary
        return result

    monkeypatch.setattr(probe.OwnedDirectory, "write", replaced)
    with pytest.raises(KeyboardInterrupt) as caught:
        run_verify(packet, handoff, retained)
    assert caught.value is primary and (packet.out / probe.FINAL_NAME).read_bytes() == b"foreign"


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failure_after_directory_release_removes_still_owned_receipt(packet, monkeypatch, exception_type):
    retained, handoff = captured(packet)
    original_close = probe.close
    primary = exception_type("post-directory-release synthetic abort")
    triggered = False

    def released(fd, error=None):
        nonlocal triggered
        observed = os.fstat(fd)
        directory = os.path.isdir(packet.out) and \
            (observed.st_dev, observed.st_ino) == \
            (packet.out.stat().st_dev, packet.out.stat().st_ino)
        original_close(fd, error)
        if directory and error is None and (packet.out / probe.FINAL_NAME).exists() and not triggered:
            triggered = True
            raise primary

    monkeypatch.setattr(probe, "close", released)
    with pytest.raises(exception_type) as caught:
        run_verify(packet, handoff, retained)
    assert caught.value is primary and triggered
    assert not (packet.out / probe.FINAL_NAME).exists()
    assert set(p.name for p in packet.out.iterdir()) == set(probe.CAPTURE_NAMES)


def test_post_release_abort_does_not_adopt_foreign_directory(packet, monkeypatch):
    retained, handoff = captured(packet)
    original_close = probe.close
    primary = KeyboardInterrupt("foreign-directory post-release abort")
    triggered = False

    def replaced(fd, error=None):
        nonlocal triggered
        observed = os.fstat(fd)
        directory = (observed.st_dev, observed.st_ino) == \
            (packet.out.stat().st_dev, packet.out.stat().st_ino)
        original_close(fd, error)
        if directory and error is None and (packet.out / probe.FINAL_NAME).exists() and not triggered:
            triggered = True
            replacement = packet.root / "after_close_foreign"
            shutil.copytree(packet.out, replacement)
            packet.out.rename(packet.root / "kept_before_release")
            replacement.rename(packet.out)
            raise primary

    monkeypatch.setattr(probe, "close", replaced)
    with pytest.raises(KeyboardInterrupt) as caught:
        run_verify(packet, handoff, retained)
    assert caught.value is primary and triggered
    assert (packet.out / probe.FINAL_NAME).exists()  # Foreign copy is preserved.


@pytest.mark.parametrize("target", ["handoff", "profile"])
def test_late_mutation_removes_owned_verification_receipt(packet, monkeypatch, target):
    retained, handoff = captured(packet)
    original = probe.OwnedDirectory.write

    def mutate(self, name, data):
        result = original(self, name, data)
        if name == probe.FINAL_NAME:
            path = handoff if target == "handoff" else packet.out / probe.CAPTURE_NAMES[0]
            contents = bytearray(path.read_bytes())
            contents[-2] ^= 1
            path.write_bytes(contents)
        return result

    monkeypatch.setattr(probe.OwnedDirectory, "write", mutate)
    with pytest.raises(probe.ProbeError):
        run_verify(packet, handoff, retained)
    assert not (packet.out / probe.FINAL_NAME).exists()


def test_handoff_running_cap_no_oversized_allocation(packet, monkeypatch):
    retained, handoff = captured(packet)
    monkeypatch.setattr(probe, "HANDOFF_LIMIT", 1)
    with pytest.raises(probe.ProbeError, match="read limit"):
        run_verify(packet, handoff, retained)


def test_receipt_stdout_cap_and_cli_mode_boundary(monkeypatch):
    with pytest.raises(probe.ProbeError, match="byte limit"):
        probe.encode({"run_id": "x" * probe.HANDOFF_LIMIT}, probe.HANDOFF_LIMIT)
    monkeypatch.setattr(probe, "capture", forbidden)
    with pytest.raises(probe.ProbeError, match="verify-only"):
        probe.main(["--mode", "capture", "--spec", "/s", "--spec-sha256", "a" * 64,
                    "--out", "/o", "--capture-receipt", "/r"])
    monkeypatch.setattr(probe, "verify", forbidden)
    with pytest.raises(probe.ProbeError, match="requires"):
        probe.main(["--mode", "verify", "--spec", "/s", "--spec-sha256", "a" * 64, "--out", "/o"])


def test_main_exact_single_line_stdout_and_verify_no_stdout(monkeypatch, capsys):
    line = b'{"synthetic":true}\n'
    monkeypatch.setattr(probe, "capture", lambda *a: line)
    probe.main(["--mode", "capture", "--spec", "/s", "--spec-sha256", "a" * 64, "--out", "/o"])
    assert capsys.readouterr().out == line.decode()
    monkeypatch.setattr(probe, "verify", lambda *a: None)
    probe.main(["--mode", "verify", "--spec", "/s", "--spec-sha256", "a" * 64, "--out", "/o",
                "--capture-receipt", "/r", "--capture-receipt-sha256", "b" * 64])
    assert capsys.readouterr().out == ""
