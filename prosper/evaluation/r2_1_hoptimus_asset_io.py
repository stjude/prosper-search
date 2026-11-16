"""Declared input custody, not a runner or an atomic filesystem snapshot."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import BinaryIO, Iterator

from .r2_1_hoptimus_io import strict_json


class AssetError(ValueError):
    """A declared input, consumption interval or CPU provenance check failed."""


def _require(condition: bool, context: str) -> None:
    if not condition:
        raise AssetError(context)


def _text(value: object, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise AssetError(f"{context}: nonempty string required")
    return value


def _digest(value: object, context: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AssetError(f"{context}: lowercase SHA256 required")
    return value


def _integer(value: object, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AssetError(f"{context}: invalid integer")
    return value


def _path_text(value: object, context: str) -> str:
    _require(isinstance(value, (str, Path)), f"{context}: explicit string/Path required")
    value = str(value)
    _require(bool(value) and "\x00" not in value, f"{context}: invalid path")
    return value


@dataclass(frozen=True, slots=True)
class AssetSpec:
    asset_id: str
    role: str
    path: str | Path
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _text(self.asset_id, "asset_id")
        _text(self.role, "role")
        object.__setattr__(self, "path", _path_text(self.path, "asset path"))
        _digest(self.sha256, self.asset_id)
        _integer(self.size, f"{self.asset_id} size")


@dataclass(frozen=True, slots=True)
class AssetPolicy:
    access: str
    encoded_limit: int

    def __post_init__(self) -> None:
        _require(type(self.access) is str and self.access in ("bytes", "path"),
                 "policy: access must be bytes or path")
        _integer(self.encoded_limit, "policy encoded_limit", 1)


@dataclass(frozen=True, slots=True)
class _Declaration:
    asset_id: str
    role: str
    path: str
    resolved_path: str
    sha256: str
    size: int


def _identity(observed: os.stat_result) -> tuple[int, int, int]:
    return observed.st_dev, observed.st_ino, observed.st_size


def _note(primary: BaseException, additional: BaseException, context: str) -> None:
    BaseException.add_note(primary, f"{context}: {type(additional).__name__}: {additional}")


class AssetGuard:
    """Serial input leases; final_bindings explicitly sweeps and seals the guard.

    Returned bytes belong to the caller. Path consumers must freeze their inputs
    cooperatively: retaining a descriptor does not bind a loader's reopened path.
    """

    def __init__(
        self, assets: Iterable[AssetSpec], *, base_dir: str | Path,
        policies: Mapping[str, AssetPolicy],
    ) -> None:
        base = Path(_path_text(base_dir, "base_dir")).absolute()
        _require(isinstance(policies, Mapping), "policies: mapping required")
        copied_policies = {}
        for role, policy in dict(policies).items():
            _text(role, "policy role")
            _require(type(policy) is AssetPolicy, f"{role}: AssetPolicy required")
            copied_policies[role] = AssetPolicy(policy.access, policy.encoded_limit)
        try:
            specifications = tuple(assets)
        except TypeError as exc:
            raise AssetError("assets: iterable of AssetSpec required") from exc
        _require(bool(specifications), "assets: nonempty declarations required")
        declarations = {}
        role_paths = set()
        aliases = {}
        for supplied in specifications:
            _require(type(supplied) is AssetSpec, "assets: AssetSpec required")
            copied = AssetSpec(supplied.asset_id, supplied.role, supplied.path,
                               supplied.sha256, supplied.size)
            _require(copied.asset_id not in declarations, "duplicate asset ID")
            _require(copied.role in copied_policies, f"{copied.asset_id}: unknown role")
            lexical = Path(copied.path)
            if not lexical.is_absolute():
                lexical = base / lexical
            try:
                resolved = lexical.resolve()
                try:
                    observed = lexical.stat()
                except FileNotFoundError:
                    observed = None  # Missing inputs are validated on acquisition.
            except (OSError, RuntimeError, ValueError) as exc:
                raise AssetError(f"{copied.asset_id}: invalid lexical path: {exc}") from exc
            role_path = (copied.role, str(resolved))
            _require(role_path not in role_paths, "duplicate role/resolved_path declaration")
            role_paths.add(role_path)
            alias_keys = [("path", str(resolved))]
            if observed is not None:
                alias_keys.append(("file", observed.st_dev, observed.st_ino))
            for key in alias_keys:
                declaration = (copied.size, copied.sha256)
                _require(key not in aliases or aliases[key] == declaration,
                         f"{copied.asset_id}: inconsistent physical-file declarations")
                aliases[key] = declaration
            declarations[copied.asset_id] = _Declaration(
                copied.asset_id, copied.role, str(lexical), str(resolved),
                copied.sha256, copied.size,
            )
        self._declarations = MappingProxyType(declarations)
        self._policies = MappingProxyType(copied_policies)
        self._observations: dict[str, tuple[int, int, int]] = {}
        self._lease_handle: BinaryIO | None = None
        self._state = "new"

    @property
    def state(self) -> str:
        return self._state

    def _fail(self) -> None:
        if self._state != "closed":
            self._state = "failed"

    def _ready(self) -> None:
        _require(self._state == "active", f"guard: requires active state, not {self._state}")
        _require(self._lease_handle is None, "guard: access/finalization during active lease")

    def __enter__(self) -> AssetGuard:
        try:
            _require(self._state == "new", "guard: enter exactly once from new")
            self._state = "active"
            return self
        except BaseException:
            self._fail()
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        primary = exc
        try:
            if exc is not None:
                self._fail()
            if self._lease_handle is not None:
                try:
                    raise AssetError("guard: context exited during active lease")
                except AssetError as additional:
                    if exc is None:
                        raise
                    _note(exc, additional, "input custody")
        except BaseException as exit_error:
            primary = exit_error
            raise
        finally:
            try:
                if self._lease_handle is not None:
                    self._close(self._lease_handle, primary)
            finally:
                self._state = "closed"
        return False

    def _entry(self, asset_id: str, access: str | None = None) -> _Declaration:
        _text(asset_id, "asset_id")
        _require(asset_id in self._declarations, f"{asset_id}: undeclared asset")
        entry = self._declarations[asset_id]
        policy = self._policies[entry.role]
        if access is not None:
            _require(policy.access == access, f"{asset_id}: forbidden {access} access")
        _require(entry.size <= policy.encoded_limit, f"{asset_id}: encoded size exceeds policy")
        return entry

    def _observe(self, entry: _Declaration, handle: BinaryIO) -> tuple[int, int, int]:
        try:
            lexical = Path(entry.path)
            _require(str(lexical.resolve(strict=True)) == entry.resolved_path,
                     f"{entry.asset_id}: changed path resolution")
            path_stat = lexical.stat()
            fd_stat = os.fstat(handle.fileno())
        except (OSError, RuntimeError, ValueError) as exc:
            raise AssetError(f"{entry.asset_id}: acquisition observation failed: {exc}") from exc
        _require(stat.S_ISREG(path_stat.st_mode) and stat.S_ISREG(fd_stat.st_mode),
                 f"{entry.asset_id}: regular file required")
        observed = _identity(fd_stat)
        _require(_identity(path_stat) == observed and fd_stat.st_size == entry.size,
                 f"{entry.asset_id}: changed acquisition identity/declared size")
        previous = self._observations.get(entry.asset_id)
        _require(previous is None or previous == observed,
                 f"{entry.asset_id}: changed same-run acquisition")
        self._observations[entry.asset_id] = observed
        return observed

    def _close(self, handle: BinaryIO, primary: BaseException | None) -> None:
        try:
            handle.close()
        except BaseException as additional:
            self._fail()
            if primary is not None:
                _note(primary, additional, "descriptor cleanup")
            else:
                raise AssetError(f"descriptor cleanup failed: {additional}") from additional

    @contextmanager
    def _acquire(self, entry: _Declaration) -> Iterator[BinaryIO]:
        try:
            lexical = Path(entry.path)
            _require(str(lexical.resolve(strict=True)) == entry.resolved_path,
                     f"{entry.asset_id}: changed path resolution")
            before = lexical.stat()
            _require(stat.S_ISREG(before.st_mode), f"{entry.asset_id}: regular file required")
            _require(before.st_size == entry.size
                     and before.st_size <= self._policies[entry.role].encoded_limit,
                     f"{entry.asset_id}: declared size/policy mismatch")
            descriptor = os.open(lexical, os.O_RDONLY | os.O_NONBLOCK)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AssetError(f"{entry.asset_id}: acquisition failed: {exc}") from exc
        try:
            handle = os.fdopen(descriptor, "rb", buffering=0)
        except BaseException as primary:
            try:
                os.close(descriptor)
            except BaseException as additional:
                _note(primary, additional, "descriptor cleanup")
            raise
        primary = None
        try:
            _require(self._observe(entry, handle) == _identity(before),
                     f"{entry.asset_id}: changed between stat and open")
            yield handle
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self._close(handle, primary)

    def _scan(self, entry: _Declaration, handle: BinaryIO, *, retain: bool) -> bytes | None:
        self._observe(entry, handle)
        hasher, total = hashlib.sha256(), 0
        buffer = io.BytesIO() if retain else None
        try:
            handle.seek(0)
            while True:
                chunk = handle.read(min(1024 * 1024, entry.size - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                _require(total <= entry.size, f"{entry.asset_id}: growth/exceeded bounded read")
                hasher.update(chunk)
                if buffer is not None:
                    buffer.write(chunk)
            _require(total == entry.size, f"{entry.asset_id}: truncation/declared size mismatch")
            self._observe(entry, handle)
            _require(hasher.hexdigest() == entry.sha256, f"{entry.asset_id}: SHA256 mismatch")
            return buffer.getvalue() if buffer is not None else None
        except OSError as exc:
            raise AssetError(f"{entry.asset_id}: descriptor read failed: {exc}") from exc
        finally:
            if buffer is not None:
                buffer.close()

    def read_verified(self, asset_id: str) -> bytes:
        try:
            self._ready()
            entry = self._entry(asset_id, "bytes")
            with self._acquire(entry) as handle:
                data = self._scan(entry, handle, retain=True)
            if type(data) is not bytes:
                raise AssetError(f"{asset_id}: immutable bytes required")
            return data
        except BaseException:
            self._fail()
            raise

    @contextmanager
    def lease_verified_path(self, asset_id: str) -> Iterator[Path]:
        try:
            self._ready()
            entry = self._entry(asset_id, "path")
            with self._acquire(entry) as handle:
                self._scan(entry, handle, retain=False)
                self._lease_handle = handle
                primary = None
                try:
                    yield Path(entry.resolved_path)
                except BaseException as exc:
                    primary = exc
                    self._fail()
                    raise
                finally:
                    try:
                        self._scan(entry, handle, retain=False)
                        if primary is None:
                            _require(self._state == "active",
                                     f"{asset_id}: guard failed during consumption")
                    except BaseException as additional:
                        self._fail()
                        if primary is not None:
                            _note(primary, additional, f"{asset_id}: post-consumption custody")
                        else:
                            raise
                    finally:
                        self._lease_handle = None
        except BaseException:
            self._fail()
            raise

    def final_bindings(self) -> tuple[dict[str, str | int], ...]:
        try:
            self._ready()
            records = []
            for asset_id in sorted(self._declarations):
                entry = self._entry(asset_id)
                with self._acquire(entry) as handle:
                    self._scan(entry, handle, retain=False)
                device, inode, _ = self._observations[asset_id]
                records.append(dict(
                    asset_id=asset_id, role=entry.role, path=entry.path,
                    resolved_path=entry.resolved_path, size=entry.size,
                    sha256=entry.sha256, device=device, inode=inode,
                ))
            self._state = "sealed"
            return tuple(records)
        except BaseException:
            self._fail()
            raise


_CPU_CONTRACT = "4b1fcb9433b2802e1545d1214050871dd8135c482f4f26959dcf5ea8c36e9d02"
_CPU_BRANCHES = MappingProxyType({
    "main": ("hoptimus_main", "hoptimus_main_saved_rank_evaluation_only"),
    "filter": ("hoptimus_filter_manifest", "hoptimus_filter_manifests_only"),
    "perturb": ("hoptimus_perturb_manifest", "hoptimus_perturb_stimulus_manifests_only"),
})


def _leaf(value: object, context: str) -> str:
    name = _text(value, context)
    _require(name not in (".", "..") and not any(c in name for c in "/\\\x00"),
             f"{context}: unsafe artifact leaf")
    return name


def _receipt_key(value: object) -> str:
    key = _text(value, "receipt_key")
    components = key.split("/")
    _require(not any(c in key for c in "\\\x00")
             and all(component not in ("", ".", "..") for component in components)
             and components[-1] == "COMPLETE.json",
             "receipt_key: safe relative COMPLETE.json path required")
    return key


def _control(data: object, context: str) -> dict:
    _require(type(data) is bytes, f"{context}: exact immutable bytes required")
    try:
        value = strict_json(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AssetError(f"{context}: invalid strict finite JSON: {exc}") from exc
    _require(type(value) is dict and type(value.get("schema_version")) is int
             and value["schema_version"] == 1, f"{context}: integer schema_version 1 required")
    return value


def verify_cpu_review_link(
    review_bytes: bytes, complete_bytes: bytes, *, branch: str, receipt_key: str,
    consumed: Mapping[str, bytes],
) -> None:
    """Check bound CPU byte provenance; the caller must custody-bind the review."""
    _require(isinstance(consumed, Mapping), "consumed: mapping required")
    snapshot = dict(consumed)
    for name, data in snapshot.items():
        _leaf(name, "consumed artifact")
        _require(type(data) is bytes, f"{name}: exact immutable bytes required")
    _require("input_bindings.json" in snapshot, "consumed: input_bindings.json required")
    _require(type(branch) is str and branch in _CPU_BRANCHES, "unknown CPU branch")
    receipt_key = _receipt_key(receipt_key)
    review = _control(review_bytes, "review")
    complete = _control(complete_bytes, "COMPLETE")
    bindings = _control(snapshot["input_bindings.json"], "input_bindings")
    _require(review.get("disposition") == "accepted", "review: disposition not accepted")
    branches = review.get("branches")
    _require(type(branches) is dict and type(branches.get(branch)) is dict
             and branches[branch].get("disposition") == "accepted",
             f"review: branch {branch} not accepted")
    _require(review.get("contract_sha256") == _CPU_CONTRACT, "review: frozen CPU contract mismatch")
    artifacts = review.get("artifacts")
    _require(type(artifacts) is dict
             and _digest(artifacts.get(receipt_key), "review COMPLETE digest")
             == hashlib.sha256(complete_bytes).hexdigest(), "review: COMPLETE link mismatch")
    task, scope = _CPU_BRANCHES[branch]
    _require(complete.get("task") == task and complete.get("scope") == scope
             and bindings.get("task") == task, "CPU branch task/scope mismatch")
    run_id = _text(complete.get("run_id"), "COMPLETE run_id")
    _require(_text(bindings.get("run_id"), "input_bindings run_id") == run_id,
             "CPU run IDs mismatch")
    spec_digest = _digest(complete.get("spec_sha256"), "COMPLETE spec digest")
    _require(_digest(bindings.get("spec_sha256"), "input_bindings spec digest") == spec_digest,
             "CPU top-level spec digests mismatch")
    entries = bindings.get("bindings")
    _require(type(entries) is list, "input_bindings: bindings list required")
    identities, spec_entries = set(), []
    for entry in entries:
        _require(type(entry) is dict, "input_bindings: binding object required")
        kind = _text(entry.get("kind"), "binding kind")
        name = _text(entry.get("name"), "binding name")
        identity = (kind, name)
        _require(identity not in identities, "input_bindings: duplicate binding identity")
        identities.add(identity)
        if kind == "spec":
            spec_entries.append(entry)
    _require(len(spec_entries) == 1 and spec_entries[0]["name"] == "spec",
             "input_bindings: exactly one spec/spec binding required")
    _require(_digest(spec_entries[0].get("sha256"), "spec binding digest") == spec_digest,
             "input_bindings: inconsistent spec binding")
    outputs = complete.get("outputs")
    _require(type(outputs) is dict, "COMPLETE: output bindings required")
    for name, entry in outputs.items():
        _leaf(name, "COMPLETE output")
        _require(type(entry) is dict, f"{name}: output binding object required")
        _digest(entry.get("sha256"), f"{name} output digest")
        _integer(entry.get("size"), f"{name} output size")
    for name, data in snapshot.items():
        _require(name in outputs, f"{name}: consumed artifact not declared by COMPLETE")
        entry = outputs[name]
        _require(entry["size"] == len(data)
                 and entry["sha256"] == hashlib.sha256(data).hexdigest(),
                 f"{name}: consumed size/hash mismatch")
