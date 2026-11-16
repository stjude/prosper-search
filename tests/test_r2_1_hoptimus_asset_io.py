"""Tiny synthetic input custody and pure CPU-receipt counterexamples."""

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path

import pytest

from prosper.evaluation import r2_1_hoptimus_asset_io as assets


def sha(data):
    return hashlib.sha256(data).hexdigest()


def spec(path, *, asset_id="asset", role="feature", data=b"original"):
    return assets.AssetSpec(asset_id, role, path, sha(data), len(data))


def make_guard(tmp_path, *, access="bytes", data=b"original"):
    path = tmp_path / "asset.bin"
    path.write_bytes(data)
    guard = assets.AssetGuard(
        [spec(path, data=data)], base_dir=tmp_path,
        policies={"feature": assets.AssetPolicy(access, 2 * 1024 * 1024)},
    )
    return guard, path


def track_handles(monkeypatch):
    opened = []
    original = assets.os.fdopen

    def tracked(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(assets.os, "fdopen", tracked)
    return opened


def assert_terminal(guard):
    assert guard.state == "failed"
    for operation in (lambda: guard.read_verified("asset"),
                      lambda: guard.lease_verified_path("asset").__enter__(),
                      guard.final_bindings, guard.__enter__):
        with pytest.raises(assets.AssetError):
            operation()
        assert guard.state == "failed"


@pytest.mark.parametrize("field,value", [
    ("asset_id", ""), ("asset_id", " "), ("asset_id", 1),
    ("role", ""), ("role", False), ("path", ""), ("path", b"a"),
    ("path", "a\x00b"), ("sha256", "A" * 64), ("sha256", "g" * 64),
    ("sha256", "a" * 63), ("sha256", None),
    ("size", True), ("size", 1.0), ("size", -1), ("size", "1"),
])
def test_invalid_asset_specs(field, value):
    fields = dict(asset_id="asset", role="feature", path="file", sha256="a" * 64, size=1)
    fields[field] = value
    with pytest.raises(assets.AssetError):
        assets.AssetSpec(**fields)


@pytest.mark.parametrize("access,limit", [
    ("auto", 1), ("BYTES", 1), (True, 1), (None, 1),
    ("bytes", 0), ("bytes", -1), ("path", True),
    ("bytes", 1.0), ("path", "2"), ("bytes", None),
])
def test_invalid_policies(access, limit):
    with pytest.raises(assets.AssetError):
        assets.AssetPolicy(access, limit)


def test_public_records_immutable_and_path_snapshot():
    declaration = spec(Path("relative.bin"))
    assert declaration.path == "relative.bin"
    with pytest.raises(FrozenInstanceError):
        declaration.size = 5
    policy = assets.AssetPolicy("bytes", 10)
    with pytest.raises(FrozenInstanceError):
        policy.access = "path"


@pytest.mark.parametrize("case", [
    "empty", "bad-record", "duplicate-id", "unknown-role", "duplicate-role-path",
    "conflicting-alias", "conflicting-hardlink", "bad-policies", "bad-policy", "bad-base",
])
def test_invalid_guard_declarations(tmp_path, case):
    path = tmp_path / "file"
    path.write_bytes(b"original")
    entries = [spec(path)]
    policies = {"feature": assets.AssetPolicy("bytes", 10),
                "other": assets.AssetPolicy("path", 10)}
    base = tmp_path
    if case == "empty":
        entries = []
    elif case == "bad-record":
        entries = [dict(asset_id="asset")]
    elif case == "duplicate-id":
        entries.append(spec(tmp_path / "second"))
    elif case == "unknown-role":
        entries = [spec(path, role="unknown")]
    elif case == "duplicate-role-path":
        link = tmp_path / "link"
        link.symlink_to(path)
        entries.append(spec(link, asset_id="second"))
    elif case in ("conflicting-alias", "conflicting-hardlink"):
        alias = tmp_path / "alias"
        if case == "conflicting-hardlink":
            os.link(path, alias)
        else:
            alias.symlink_to(path)
        entries.append(spec(alias, asset_id="second", role="other", data=b"different"))
    elif case == "bad-policies":
        policies = []
    elif case == "bad-policy":
        policies["feature"] = {"access": "bytes", "encoded_limit": 10}
    else:
        base = 1
    with pytest.raises(assets.AssetError):
        assets.AssetGuard(entries, base_dir=base, policies=policies)


def test_private_value_snapshots_and_anchored_cwd(tmp_path, monkeypatch):
    origin, other = tmp_path / "origin", tmp_path / "other"
    origin.mkdir()
    other.mkdir()
    (origin / "asset.bin").write_bytes(b"original")
    (other / "asset.bin").write_bytes(b"modified")
    monkeypatch.chdir(tmp_path)
    supplied = spec("asset.bin")
    policy = assets.AssetPolicy("bytes", 8)
    entries, policies = [supplied], {"feature": policy}
    guard = assets.AssetGuard(entries, base_dir=Path("origin"), policies=policies)
    entries.clear()
    policies.clear()
    object.__setattr__(supplied, "path", str(other / "asset.bin"))
    object.__setattr__(supplied, "sha256", sha(b"modified"))
    object.__setattr__(supplied, "asset_id", "retargeted")
    object.__setattr__(supplied, "role", "unknown")
    object.__setattr__(supplied, "size", 0)
    object.__setattr__(policy, "access", "path")
    object.__setattr__(policy, "encoded_limit", 1)
    monkeypatch.chdir(other)
    with guard:
        assert guard.read_verified("asset") == b"original"
        binding, = guard.final_bindings()
        assert binding["path"] == str(origin / "asset.bin")
        assert binding["sha256"] == sha(b"original")


def test_construction_is_metadata_only(tmp_path, monkeypatch):
    path = tmp_path / "asset"
    path.write_bytes(b"modified")
    monkeypatch.setattr(assets.os, "open", lambda *a, **k: pytest.fail("construction reads no bytes"))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("construction reads no bytes"))
    guard = assets.AssetGuard([spec(path)], base_dir=tmp_path,
                              policies={"feature": assets.AssetPolicy("bytes", 8)})
    assert guard.state == "new"
    with guard:
        pass
    assert guard.state == "closed"


def test_lexical_symlink_base(tmp_path):
    lexical, physical = tmp_path / "lexical", tmp_path / "physical"
    lexical.mkdir()
    physical.mkdir()
    (lexical / "asset.bin").write_bytes(b"original")
    (physical / "declaration.json").write_bytes(b"{}")
    declaration = lexical / "declaration.json"
    declaration.symlink_to(physical / "declaration.json")
    guard = assets.AssetGuard([spec("asset.bin")], base_dir=declaration.absolute().parent,
                              policies={"feature": assets.AssetPolicy("bytes", 8)})
    with guard:
        assert guard.read_verified("asset") == b"original"
        record, = guard.final_bindings()
        assert record["path"] == str(lexical / "asset.bin")
        assert record["resolved_path"] == str(lexical / "asset.bin")


def test_symlink_dotdot_not_lexically_collapsed(tmp_path):
    lexical, physical = tmp_path / "lexical", tmp_path / "physical"
    lexical.mkdir()
    physical.mkdir()
    (physical / "inside").mkdir()
    (lexical / "link").symlink_to(physical / "inside", target_is_directory=True)
    (lexical / "asset.bin").write_bytes(b"modified")
    (physical / "asset.bin").write_bytes(b"original")
    guard = assets.AssetGuard([spec("link/../asset.bin")], base_dir=lexical,
                              policies={"feature": assets.AssetPolicy("bytes", 8)})
    with guard:
        assert guard.read_verified("asset") == b"original"
        record, = guard.final_bindings()
        assert record["path"] == str(lexical / "link" / ".." / "asset.bin")
        assert record["resolved_path"] == str(physical / "asset.bin")


def test_symlink_dotdot_in_lexical_base(tmp_path):
    lexical, physical = tmp_path / "lexical", tmp_path / "physical"
    lexical.mkdir()
    physical.mkdir()
    (physical / "inside").mkdir()
    (lexical / "link").symlink_to(physical / "inside", target_is_directory=True)
    (physical / "asset.bin").write_bytes(b"original")
    guard = assets.AssetGuard([spec("asset.bin")], base_dir=lexical / "link" / "..",
                              policies={"feature": assets.AssetPolicy("bytes", 8)})
    with guard:
        assert guard.read_verified("asset") == b"original"
        binding, = guard.final_bindings()
        assert binding["path"] == str(lexical / "link" / ".." / "asset.bin")
        assert binding["resolved_path"] == str(physical / "asset.bin")


@pytest.mark.parametrize("size", [0, 8, 1024 * 1024 + 1, 2 * 1024 * 1024])
def test_verified_bytes_chunk_bounds_and_descriptor_cleanup(tmp_path, monkeypatch, size):
    data = b"x" * size
    guard, path = make_guard(tmp_path, data=data)
    opened = track_handles(monkeypatch)
    read_sizes = []
    original = assets.os.fdopen

    class RecordingHandle:
        def __init__(self, handle):
            self.handle = handle

        def read(self, count):
            read_sizes.append(count)
            return self.handle.read(count)

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(assets.os, "fdopen",
                        lambda *a, **k: RecordingHandle(original(*a, **k)))
    with guard:
        result = guard.read_verified("asset")
        assert type(result) is bytes and result == data
        assert all(handle.closed for handle in opened)
        binding, = guard.final_bindings()
        observed = path.stat()
        assert binding == dict(asset_id="asset", role="feature", path=str(path),
                               resolved_path=str(path), size=size, sha256=sha(data),
                               device=observed.st_dev, inode=observed.st_ino)
        assert guard.state == "sealed"
    assert guard.state == "closed"
    assert max(read_sizes) <= 1024 * 1024
    assert all(handle.closed for handle in opened)


def test_stable_binding_order_and_distinct_roles(tmp_path):
    path = tmp_path / "shared"
    path.write_bytes(b"original")
    guard = assets.AssetGuard(
        [spec(path, asset_id="z"), spec(path, asset_id="a", role="checkpoint")],
        base_dir=tmp_path,
        policies={"feature": assets.AssetPolicy("bytes", 8),
                  "checkpoint": assets.AssetPolicy("path", 8)},
    )
    with guard:
        assert guard.read_verified("z") == b"original"
        with guard.lease_verified_path("a") as leased:
            assert leased == path
        records = guard.final_bindings()
        assert [record["asset_id"] for record in records] == ["a", "z"]


def test_consistent_hardlink_roles(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"original")
    os.link(first, second)
    guard = assets.AssetGuard(
        [spec(first), spec(second, asset_id="second", role="checkpoint")], base_dir=tmp_path,
        policies={"feature": assets.AssetPolicy("bytes", 8),
                  "checkpoint": assets.AssetPolicy("path", 8)},
    )
    with guard:
        guard.read_verified("asset")
        with guard.lease_verified_path("second"):
            pass
        records = guard.final_bindings()
        assert records[0]["inode"] == records[1]["inode"]


@pytest.mark.parametrize("asset_id", ["unknown", "", None, True, 1, []])
def test_invalid_access_identity_terminal(tmp_path, asset_id):
    guard, _ = make_guard(tmp_path)
    with guard:
        with pytest.raises(assets.AssetError):
            guard.read_verified(asset_id)
        assert_terminal(guard)


@pytest.mark.parametrize("access,method", [("path", "bytes"), ("bytes", "path")])
def test_access_policy_failures_terminal(tmp_path, access, method):
    guard, _ = make_guard(tmp_path, access=access)
    with guard:
        with pytest.raises(assets.AssetError, match="forbidden"):
            if method == "bytes":
                guard.read_verified("asset")
            else:
                with guard.lease_verified_path("asset"):
                    pytest.fail("must not yield")
        assert_terminal(guard)


@pytest.mark.parametrize("case", ["size", "digest", "missing", "directory", "fifo", "symlink"])
def test_acquisition_failures_and_restoration_are_terminal(tmp_path, monkeypatch, case):
    guard, path = make_guard(tmp_path)
    opened = track_handles(monkeypatch)
    if case == "size":
        path.write_bytes(b"short")
    elif case == "digest":
        path.write_bytes(b"modified")
    elif case == "missing":
        path.unlink()
    elif case == "directory":
        path.unlink()
        path.mkdir()
    elif case == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        target = tmp_path / "same-bytes"
        target.write_bytes(b"original")
        path.unlink()
        path.symlink_to(target)
    with guard:
        with pytest.raises(assets.AssetError):
            guard.read_verified("asset")
        if path.is_dir():
            path.rmdir()
        elif path.exists() or path.is_symlink():
            path.unlink()
        path.write_bytes(b"original")
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


def test_large_limits_by_declaration_without_large_payload(tmp_path, monkeypatch):
    declaration = assets.AssetSpec("asset", "checkpoint", "never-read", "a" * 64, 4539151464)
    guard = assets.AssetGuard([declaration], base_dir=tmp_path,
                              policies={"checkpoint": assets.AssetPolicy("path", 512 * 1024**2)})
    monkeypatch.setattr(assets.os, "open", lambda *a, **k: pytest.fail("must reject before open"))
    with guard:
        with pytest.raises(assets.AssetError, match="encoded size"):
            with guard.lease_verified_path("asset"):
                pytest.fail("must not yield")
        assert_terminal(guard)
    small, _ = make_guard(tmp_path, access="path")
    assert assets.AssetPolicy("path", 4539151464).encoded_limit == 4539151464
    assert small.state == "new"


def test_mutation_between_stat_and_open(tmp_path, monkeypatch):
    guard, path = make_guard(tmp_path)
    opened = track_handles(monkeypatch)
    original = assets.os.open

    def substituted(name, flags):
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"original")
        replacement.replace(path)
        return original(name, flags)

    monkeypatch.setattr(assets.os, "open", substituted)
    with guard:
        with pytest.raises(assets.AssetError, match="between stat and open"):
            guard.read_verified("asset")
        assert_terminal(guard)
    assert opened and all(handle.closed for handle in opened)


@pytest.mark.parametrize("mutation", ["growth", "truncate", "same-size", "replacement"])
def test_mutation_while_descriptor_reading(tmp_path, monkeypatch, mutation):
    guard, path = make_guard(tmp_path)
    opened = track_handles(monkeypatch)
    original = assets.os.fdopen
    mutated = False

    class MutatingHandle:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size):
            nonlocal mutated
            chunk = self.handle.read(size)
            if not mutated:
                mutated = True
                if mutation == "growth":
                    path.write_bytes(b"original+")
                elif mutation == "truncate":
                    path.write_bytes(b"x")
                elif mutation == "same-size":
                    self.handle.seek(0)
                    path.write_bytes(b"modified")
                else:
                    replacement = tmp_path / "replacement"
                    replacement.write_bytes(b"original")
                    replacement.replace(path)
            return chunk

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(assets.os, "fdopen",
                        lambda *a, **k: MutatingHandle(original(*a, **k)))
    with guard:
        with pytest.raises(assets.AssetError):
            guard.read_verified("asset")
        path.write_bytes(b"original")
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


@pytest.mark.parametrize("mutation", ["digest", "replacement", "symlink"])
def test_final_sweep_detects_late_changes(tmp_path, monkeypatch, mutation):
    guard, path = make_guard(tmp_path)
    opened = track_handles(monkeypatch)
    with guard:
        assert guard.read_verified("asset") == b"original"
        if mutation == "digest":
            path.write_bytes(b"modified")
        else:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"original")
            if mutation == "replacement":
                replacement.replace(path)
            else:
                path.unlink()
                path.symlink_to(replacement)
        with pytest.raises(assets.AssetError):
            guard.final_bindings()
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


def test_unread_asset_is_in_final_sweep(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.write_bytes(b"original")
    bad.write_bytes(b"modified")
    guard = assets.AssetGuard([spec(good), spec(bad, asset_id="second")], base_dir=tmp_path,
                              policies={"feature": assets.AssetPolicy("bytes", 8)})
    with guard:
        assert guard.read_verified("asset") == b"original"
        with pytest.raises(assets.AssetError, match="SHA256"):
            guard.final_bindings()
        assert_terminal(guard)


def test_mutation_during_final_sweep(tmp_path, monkeypatch):
    guard, path = make_guard(tmp_path)
    opened = track_handles(monkeypatch)
    original = guard._scan
    with guard:
        guard.read_verified("asset")

        def changed(entry, handle, *, retain):
            assert retain is False
            path.write_bytes(b"modified")
            return original(entry, handle, retain=retain)

        monkeypatch.setattr(guard, "_scan", changed)
        with pytest.raises(assets.AssetError, match="SHA256"):
            guard.final_bindings()
        path.write_bytes(b"original")
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


def test_verified_reads_do_not_cache_old_bytes(tmp_path):
    guard, path = make_guard(tmp_path)
    with guard:
        original = guard.read_verified("asset")
        path.write_bytes(b"modified")
        with pytest.raises(assets.AssetError, match="SHA256"):
            guard.read_verified("asset")
        assert original == b"original"
        assert_terminal(guard)


@pytest.mark.parametrize("operation", ["read", "lease", "finalize", "enter"])
def test_lease_rejects_overlapping_operations(tmp_path, operation):
    guard, path = make_guard(tmp_path, access="path")
    with guard:
        with pytest.raises(assets.AssetError, match="guard failed during consumption"):
            with guard.lease_verified_path("asset") as leased:
                assert leased == path
                with pytest.raises(assets.AssetError):
                    if operation == "read":
                        guard.read_verified("asset")
                    elif operation == "lease":
                        with guard.lease_verified_path("asset"):
                            pytest.fail("no overlapping lease")
                    elif operation == "finalize":
                        guard.final_bindings()
                    else:
                        guard.__enter__()
        assert_terminal(guard)


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("mutate", [False, True])
def test_consumer_exception_identity_integrity_notes_and_closure(
    tmp_path, monkeypatch, exception_type, mutate,
):
    guard, path = make_guard(tmp_path, access="path")
    opened = track_handles(monkeypatch)
    primary = exception_type("consumer abort")
    with guard:
        with pytest.raises(exception_type) as caught:
            with guard.lease_verified_path("asset"):
                assert len(opened) == 1 and not opened[0].closed
                if mutate:
                    path.write_bytes(b"modified")
                raise primary
        assert caught.value is primary
        notes = getattr(primary, "__notes__", [])
        assert bool(notes) is mutate
        if mutate:
            assert any("post-consumption custody" in note and "SHA256" in note for note in notes)
        path.write_bytes(b"original")
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


@pytest.mark.parametrize("mutation", ["digest", "growth", "truncate", "replacement", "symlink"])
def test_normal_lease_exit_checks_source(tmp_path, monkeypatch, mutation):
    guard, path = make_guard(tmp_path, access="path")
    opened = track_handles(monkeypatch)
    with guard:
        with pytest.raises(assets.AssetError):
            with guard.lease_verified_path("asset"):
                if mutation == "digest":
                    path.write_bytes(b"modified")
                elif mutation == "growth":
                    path.write_bytes(b"original+")
                elif mutation == "truncate":
                    path.write_bytes(b"x")
                else:
                    replacement = tmp_path / "replacement"
                    replacement.write_bytes(b"original")
                    if mutation == "replacement":
                        replacement.replace(path)
                    else:
                        path.unlink()
                        path.symlink_to(replacement)
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


@pytest.mark.parametrize("data", [b"modified", b"short"])
def test_unverified_path_never_yielded(tmp_path, monkeypatch, data):
    guard, path = make_guard(tmp_path, access="path")
    opened = track_handles(monkeypatch)
    path.write_bytes(data)
    with guard:
        with pytest.raises(assets.AssetError):
            with guard.lease_verified_path("asset"):
                pytest.fail("unverified path must not be yielded")
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


def test_guard_exit_with_live_lease_closes_descriptor(tmp_path, monkeypatch):
    guard, _ = make_guard(tmp_path, access="path")
    opened = track_handles(monkeypatch)
    guard.__enter__()
    lease = guard.lease_verified_path("asset")
    lease.__enter__()
    with pytest.raises(assets.AssetError, match="context exited during active lease"):
        guard.__exit__(None, None, None)
    assert guard.state == "closed" and all(handle.closed for handle in opened)
    with pytest.raises(assets.AssetError):
        lease.__exit__(None, None, None)
    assert guard.state == "closed"


@pytest.mark.parametrize("state", ["new", "sealed", "closed"])
@pytest.mark.parametrize("operation", ["read", "lease", "finalize"])
def test_invalid_lifecycle_states(tmp_path, state, operation):
    guard, _ = make_guard(tmp_path)
    if state != "new":
        guard.__enter__()
        guard.final_bindings()
        if state == "closed":
            guard.__exit__(None, None, None)
    with pytest.raises(assets.AssetError, match="requires active"):
        if operation == "read":
            guard.read_verified("asset")
        elif operation == "lease":
            with guard.lease_verified_path("asset"):
                pytest.fail("must not yield")
        else:
            guard.final_bindings()
    if state == "closed":
        assert guard.state == "closed"
    else:
        assert_terminal(guard)
    guard.__exit__(None, None, None)
    assert guard.state == "closed"


def test_guard_context_preserves_outer_abort(tmp_path):
    guard, _ = make_guard(tmp_path)
    primary = KeyboardInterrupt("outer abort")
    with pytest.raises(KeyboardInterrupt) as caught:
        with guard:
            guard.read_verified("asset")
            raise primary
    assert caught.value is primary and guard.state == "closed"


def test_descriptor_closed_on_fdopen_failure(tmp_path, monkeypatch):
    guard, _ = make_guard(tmp_path)
    descriptors = []
    original = assets.os.open

    def opened(*args, **kwargs):
        descriptor = original(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    primary = SystemExit("fdopen abort")
    monkeypatch.setattr(assets.os, "open", opened)
    monkeypatch.setattr(assets.os, "fdopen", lambda *a, **k: (_ for _ in ()).throw(primary))
    with guard:
        with pytest.raises(SystemExit) as caught:
            guard.read_verified("asset")
        assert caught.value is primary
        assert_terminal(guard)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_abort_during_descriptor_read_preserved(tmp_path, monkeypatch, exception_type):
    guard, _ = make_guard(tmp_path)
    opened = track_handles(monkeypatch)
    original = assets.os.fdopen
    primary = exception_type("descriptor read abort")

    class AbortingHandle:
        def __init__(self, handle):
            self.handle = handle

        def read(self, count):
            raise primary

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(assets.os, "fdopen",
                        lambda *a, **k: AbortingHandle(original(*a, **k)))
    with guard:
        with pytest.raises(exception_type) as caught:
            guard.read_verified("asset")
        assert caught.value is primary
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_cleanup_failure_attached_without_masking_abort(tmp_path, monkeypatch, exception_type):
    guard, path = make_guard(tmp_path, access="path")
    opened = track_handles(monkeypatch)
    original = assets.os.fdopen
    primary = exception_type("leased consumer abort")

    class CleanupFailureHandle:
        def __init__(self, handle):
            self.handle = handle

        def close(self):
            self.handle.close()
            raise OSError("synthetic close diagnostic")

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(assets.os, "fdopen",
                        lambda *a, **k: CleanupFailureHandle(original(*a, **k)))
    with guard:
        with pytest.raises(exception_type) as caught:
            with guard.lease_verified_path("asset"):
                path.write_bytes(b"modified")
                raise primary
        assert caught.value is primary
        assert any("post-consumption custody" in note for note in primary.__notes__)
        assert any("descriptor cleanup" in note for note in primary.__notes__)
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


def test_close_failure_without_primary_is_contextual(tmp_path, monkeypatch):
    guard, _ = make_guard(tmp_path)
    opened = track_handles(monkeypatch)
    original = assets.os.fdopen

    class CleanupFailureHandle:
        def __init__(self, handle):
            self.handle = handle

        def close(self):
            self.handle.close()
            raise OSError("synthetic close diagnostic")

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(assets.os, "fdopen",
                        lambda *a, **k: CleanupFailureHandle(original(*a, **k)))
    with guard:
        with pytest.raises(assets.AssetError, match="descriptor cleanup failed"):
            guard.read_verified("asset")
        assert_terminal(guard)
    assert all(handle.closed for handle in opened)


def encode(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


TASKS = {
    "main": ("hoptimus_main", "hoptimus_main_saved_rank_evaluation_only"),
    "filter": ("hoptimus_filter_manifest", "hoptimus_filter_manifests_only"),
    "perturb": ("hoptimus_perturb_manifest", "hoptimus_perturb_stimulus_manifests_only"),
}


def receipt_fixture(branch="main"):
    task, scope = TASKS[branch]
    binding = dict(schema_version=1, run_id="synthetic", task=task, spec_sha256="a" * 64,
                   bindings=[dict(kind="spec", name="spec", sha256="a" * 64),
                             dict(kind="source", name="metadata", sha256="b" * 64)])
    consumed = {"input_bindings.json": encode(binding), "summary.json": b'{"synthetic":true}\n'}
    complete = dict(schema_version=1, run_id="synthetic", task=task, scope=scope,
                    spec_sha256="a" * 64, outputs={})
    review = dict(
        schema_version=1, disposition="accepted",
        contract_sha256="4b1fcb9433b2802e1545d1214050871dd8135c482f4f26959dcf5ea8c36e9d02",
        branches={branch: {"disposition": "accepted"}}, artifacts={},
    )
    return dict(branch=branch, key=f"{branch}_attempt/COMPLETE.json", binding=binding,
                consumed=consumed, complete=complete, review=review)


def rebound(fixture, *, binding=True, outputs=True, receipt=True):
    if binding:
        fixture["consumed"]["input_bindings.json"] = encode(fixture["binding"])
    if outputs:
        fixture["complete"]["outputs"] = {
            name: {"sha256": sha(data), "size": len(data)}
            for name, data in fixture["consumed"].items()
        }
    complete = encode(fixture["complete"])
    if receipt:
        fixture["review"]["artifacts"] = {fixture["key"]: sha(complete)}
    return encode(fixture["review"]), complete


def verify(fixture, **rebinding):
    review, complete = rebound(fixture, **rebinding)
    return assets.verify_cpu_review_link(review, complete, branch=fixture["branch"],
                                         receipt_key=fixture["key"], consumed=fixture["consumed"])


@pytest.mark.parametrize("branch", TASKS)
def test_valid_pure_cpu_chain_and_unconsumed_outputs(branch, monkeypatch):
    fixture = receipt_fixture(branch)
    rebound(fixture)
    fixture["complete"]["outputs"]["unconsumed.bin"] = {"sha256": "c" * 64, "size": 4539151464}
    monkeypatch.setattr(assets.os, "open", lambda *a, **k: pytest.fail("pure validator opens nothing"))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("pure validator opens nothing"))
    assert verify(fixture, outputs=False) is None


def test_consumed_mapping_is_snapshotted_before_validation(monkeypatch):
    fixture = receipt_fixture()
    review, complete = rebound(fixture)
    original = assets._control
    mutated = False

    def changed(data, context):
        nonlocal mutated
        if not mutated:
            mutated = True
            fixture["consumed"].clear()
            fixture["consumed"]["bad"] = bytearray(b"mutable")
        return original(data, context)

    monkeypatch.setattr(assets, "_control", changed)
    assert assets.verify_cpu_review_link(review, complete, branch="main",
                                        receipt_key=fixture["key"],
                                        consumed=fixture["consumed"]) is None


def test_historical_device_inode_not_compared_across_nodes():
    fixture = receipt_fixture()
    for entry in fixture["binding"]["bindings"]:
        entry["device"], entry["inode"] = 999999, 777777
    assert verify(fixture) is None


def test_consumed_bytes_subclasses_rejected():
    class DerivedBytes(bytes):
        pass

    fixture = receipt_fixture()
    review, complete = rebound(fixture)
    fixture["consumed"]["summary.json"] = DerivedBytes(fixture["consumed"]["summary.json"])
    with pytest.raises(assets.AssetError, match="immutable bytes"):
        assets.verify_cpu_review_link(review, complete, branch="main",
                                      receipt_key=fixture["key"], consumed=fixture["consumed"])


@pytest.mark.parametrize("control", ["review", "complete", "binding"])
@pytest.mark.parametrize("version", [True, 1.0, 0, 2, None, "1"])
def test_schema_versions_are_not_equality_coerced(control, version):
    fixture = receipt_fixture()
    fixture[control]["schema_version"] = version
    with pytest.raises(assets.AssetError, match="schema_version"):
        verify(fixture)


@pytest.mark.parametrize("case", [
    "review-rejected", "branch-rejected", "missing-branch", "contract", "receipt-hash",
    "task", "scope", "binding-task", "run", "empty-run", "numeric-run",
    "spec-header", "spec-uppercase", "spec-record", "duplicate-spec",
    "duplicate-source", "missing-spec", "extra-spec", "spec-name",
    "invalid-binding", "missing-bindings", "binding-name", "unknown-branch",
])
def test_cpu_provenance_counterexamples(case):
    fixture = receipt_fixture()
    output_args = {}
    if case == "review-rejected":
        fixture["review"]["disposition"] = "pending"
    elif case == "branch-rejected":
        fixture["review"]["branches"]["main"]["disposition"] = "pending"
    elif case == "missing-branch":
        fixture["review"]["branches"] = {}
    elif case == "contract":
        fixture["review"]["contract_sha256"] = "b" * 64
    elif case == "receipt-hash":
        rebound(fixture)
        fixture["review"]["artifacts"][fixture["key"]] = "b" * 64
        output_args["receipt"] = False
    elif case == "task":
        fixture["complete"]["task"] = "hoptimus_filter_manifest"
    elif case == "scope":
        fixture["complete"]["scope"] = "fitting complete"
    elif case == "binding-task":
        fixture["binding"]["task"] = "hoptimus_filter_manifest"
    elif case == "run":
        fixture["binding"]["run_id"] = "other"
    elif case in ("empty-run", "numeric-run"):
        value = "" if case == "empty-run" else 123
        fixture["complete"]["run_id"] = fixture["binding"]["run_id"] = value
    elif case == "spec-header":
        fixture["binding"]["spec_sha256"] = "b" * 64
    elif case == "spec-uppercase":
        fixture["complete"]["spec_sha256"] = fixture["binding"]["spec_sha256"] = "A" * 64
    elif case == "spec-record":
        fixture["binding"]["bindings"][0]["sha256"] = "b" * 64
    elif case == "duplicate-spec":
        fixture["binding"]["bindings"].append(dict(fixture["binding"]["bindings"][0]))
    elif case == "duplicate-source":
        fixture["binding"]["bindings"].append(dict(fixture["binding"]["bindings"][1]))
    elif case == "missing-spec":
        fixture["binding"]["bindings"] = fixture["binding"]["bindings"][1:]
    elif case == "extra-spec":
        fixture["binding"]["bindings"].append(dict(kind="spec", name="other", sha256="a" * 64))
    elif case == "spec-name":
        fixture["binding"]["bindings"][0]["name"] = "other"
    elif case == "invalid-binding":
        fixture["binding"]["bindings"].append(True)
    elif case == "missing-bindings":
        del fixture["binding"]["bindings"]
    elif case == "binding-name":
        fixture["binding"]["bindings"][1]["name"] = []
    else:
        fixture["branch"] = "other"
    with pytest.raises(assets.AssetError):
        verify(fixture, **output_args)


@pytest.mark.parametrize("key", [
    "", "/COMPLETE.json", "../COMPLETE.json", "./COMPLETE.json", "a//COMPLETE.json",
    "a/../COMPLETE.json", "a/./COMPLETE.json", "a\\COMPLETE.json",
    "a/COMPLETE.json/", "a/summary.json", "a\x00/COMPLETE.json",
])
def test_unsafe_receipt_keys(key):
    fixture = receipt_fixture()
    fixture["key"] = key
    with pytest.raises(assets.AssetError, match="receipt_key"):
        verify(fixture)


@pytest.mark.parametrize("name", ["", ".", "..", "../x", "/x", "a/x", "a\\x", "a\x00x"])
@pytest.mark.parametrize("consumed", [False, True])
def test_unsafe_artifact_names_even_when_not_consumed(name, consumed):
    fixture = receipt_fixture()
    rebound(fixture)
    if consumed:
        fixture["consumed"][name] = b"x"
    else:
        fixture["complete"]["outputs"][name] = dict(size=1, sha256=sha(b"x"))
    with pytest.raises(assets.AssetError):
        verify(fixture, outputs=consumed)


@pytest.mark.parametrize("size", [True, 1.0, -1, "1", None])
def test_all_output_sizes_strict_even_unconsumed(size):
    fixture = receipt_fixture()
    rebound(fixture)
    fixture["complete"]["outputs"]["unconsumed.bin"] = dict(size=size, sha256="b" * 64)
    with pytest.raises(assets.AssetError, match="invalid integer"):
        verify(fixture, outputs=False)


@pytest.mark.parametrize("digest", ["A" * 64, "g" * 64, "a", True, None])
def test_all_output_digests_strict_even_unconsumed(digest):
    fixture = receipt_fixture()
    rebound(fixture)
    fixture["complete"]["outputs"]["unconsumed.bin"] = dict(size=0, sha256=digest)
    with pytest.raises(assets.AssetError, match="SHA256"):
        verify(fixture, outputs=False)


@pytest.mark.parametrize("case", ["missing-input-bindings", "extra", "missing-output", "size", "hash"])
def test_consumed_binding_failures(case):
    fixture = receipt_fixture()
    rebound(fixture)
    if case == "missing-input-bindings":
        del fixture["consumed"]["input_bindings.json"]
    elif case == "extra":
        fixture["consumed"]["extra.bin"] = b"extra"
    elif case == "missing-output":
        del fixture["complete"]["outputs"]["summary.json"]
    elif case == "size":
        fixture["complete"]["outputs"]["summary.json"]["size"] += 1
    else:
        fixture["complete"]["outputs"]["summary.json"]["sha256"] = "b" * 64
    with pytest.raises(assets.AssetError):
        verify(fixture, binding=case != "missing-input-bindings", outputs=False)


@pytest.mark.parametrize("value", [bytearray(b"{}"), memoryview(b"{}"), "{}", None])
@pytest.mark.parametrize("location", ["review", "complete", "consumed"])
def test_only_exact_immutable_bytes(location, value):
    fixture = receipt_fixture()
    review, complete = rebound(fixture)
    if location == "review":
        review = value
    elif location == "complete":
        complete = value
    else:
        fixture["consumed"]["summary.json"] = value
    with pytest.raises(assets.AssetError, match="immutable bytes"):
        assets.verify_cpu_review_link(review, complete, branch="main",
                                      receipt_key=fixture["key"], consumed=fixture["consumed"])


@pytest.mark.parametrize("malformed", [
    b"{", b"[]", b"null", b"\xff", b'{"schema_version":1,"schema_version":1}',
    b'{"schema_version":1,"extra":NaN}', b'{"schema_version":1,"extra":Infinity}',
    b'{"schema_version":1,"extra":1e999}', b'{"schema_version":1}{}',
])
@pytest.mark.parametrize("location", ["review", "complete", "binding"])
def test_strict_json_controls(location, malformed):
    fixture = receipt_fixture()
    if location == "binding":
        fixture["consumed"]["input_bindings.json"] = malformed
        review, complete = rebound(fixture, binding=False)
    else:
        review, complete = rebound(fixture)
        if location == "review":
            review = malformed
        else:
            complete = malformed
            fixture["review"]["artifacts"][fixture["key"]] = sha(complete)
            review = encode(fixture["review"])
    with pytest.raises(assets.AssetError):
        assets.verify_cpu_review_link(review, complete, branch="main",
                                      receipt_key=fixture["key"], consumed=fixture["consumed"])
