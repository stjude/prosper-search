"""Test selection for the released repository (see docs/TESTING.md).

Tests listed in tests/skip_registry.tsv and revision/skip_registry.tsv are marked and,
where their requirement is not met here, skipped with the reason given in the registry:

* ``site``     needs restricted data, a documented placeholder root, the executed
               workspace layout or a held record; runs only with PROSPER_SITE_TESTS=1.
* ``pinned``   needs the library versions of the executed environment; runs when the
               installed versions match the ``requires`` column.
* ``optional`` needs an optional dependency; runs when it is installed.
* ``slow``     long-running extraction tests; run with PROSPER_SLOW_TESTS=1.
* ``isolated`` needs a fresh interpreter; runs when its file is the only pytest argument.
* ``script``   a standalone evidence check, not a pytest module; never collected.

Module-level entries are skipped before the module is imported. This file does not
import prosper.
"""
import csv
import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path

import pytest

# TensorFlow is not a PROSPER dependency. If it is installed, POT imports it and its
# start-up warnings on stderr break tests that require empty stderr.
os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")

ROOT = Path(__file__).resolve().parent
REGISTRIES = (ROOT / "tests" / "skip_registry.tsv", ROOT / "revision" / "skip_registry.tsv")
MARKERS = {
    "site": "needs restricted data, a placeholder root, the executed workspace layout or a held record",
    "pinned": "needs the library versions of the executed environment",
    "optional": "needs an optional dependency",
    "slow": "long-running; set PROSPER_SLOW_TESTS=1",
    "isolated": "must run in its own pytest process",
    "script": "standalone evidence check, not collected by pytest",
}


def _load():
    entries = {}
    for path in REGISTRIES:
        if path.is_file():
            with open(path, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    entries[row["target"]] = row
    return entries


ENTRIES = _load()


def _unmet(requires):
    """Return a description of the first unmet requirement, or None."""
    for token in filter(None, requires.split(";")):
        kind, _, spec = token.partition(":")
        if kind == "versions":
            for pin in spec.split(","):
                name, _, wanted = pin.partition("==")
                try:
                    have = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    have = "not installed"
                if have != wanted:
                    return f"{name}=={wanted} required, found {have}"
        elif kind == "dist":
            try:
                importlib.metadata.version(spec)
            except importlib.metadata.PackageNotFoundError:
                return f"distribution {spec} not installed"
        elif kind == "module":
            if importlib.util.find_spec(spec) is None:
                return f"module {spec} not installed"
        elif kind == "cmd":
            if shutil.which(spec) is None:
                return f"command {spec} not on PATH"
        elif kind == "env":
            if not os.environ.get(spec):
                return f"environment variable {spec} not set"
        elif kind == "path":
            if not Path(spec).exists():
                return f"{spec} not present"
        else:
            raise ValueError(f"unknown requirement kind in skip registry: {token}")
    return None


def _skip_reason(entry, config):
    marker = entry["marker"]
    if marker == "site":
        if os.environ.get("PROSPER_SITE_TESTS") != "1":
            return f"site ({entry['category']}): {entry['reason']}; set PROSPER_SITE_TESTS=1"
        return None
    if marker == "slow":
        return None if os.environ.get("PROSPER_SLOW_TESTS") == "1" else entry["reason"]
    if marker == "script":
        return entry["reason"]
    if marker == "isolated":
        target = (ROOT / entry["target"]).resolve()
        args = [Path(a.split("::")[0]).resolve() for a in config.args]
        return None if args and all(a == target for a in args) else entry["reason"]
    missing = _unmet(entry["requires"])
    if missing is None:
        return None
    return f"{entry['category']}: {missing} ({entry['reason']})"


def pytest_configure(config):
    for name, text in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {text}")


class _SkippedModule(pytest.Module):
    skip_reason = ""

    def collect(self):
        pytest.skip(self.skip_reason, allow_module_level=True)


def pytest_pycollect_makemodule(module_path, parent):
    try:
        rel = Path(module_path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    entry = ENTRIES.get(rel)
    if entry is None:
        return None
    reason = _skip_reason(entry, parent.config)
    if reason is None:
        return None
    module = _SkippedModule.from_parent(parent, path=Path(module_path))
    module.skip_reason = reason
    return module


def pytest_collection_modifyitems(config, items):
    for item in items:
        entry = ENTRIES.get(item.nodeid)
        if entry is None:
            continue
        item.add_marker(getattr(pytest.mark, entry["marker"]))
        reason = _skip_reason(entry, config)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
