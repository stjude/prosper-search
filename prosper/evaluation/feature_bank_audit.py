"""Read-only binding audit for trusted, local legacy feature-bank pickles.

This validates *declared* identities and coordinates, not encoder provenance,
historical extraction, matrix construction, or scientific performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import pickletools
import platform
import socket
import stat
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class AuditError(ValueError):
    """An input, output, or binding failed validation."""


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _time():
    return datetime.now(timezone.utc).isoformat()


def _object_pairs(pairs):
    obj = {}
    for key, value in pairs:
        _require(key not in obj, f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _bad_constant(value):
    raise AuditError(f"invalid JSON constant: {value}")


def _json(data):
    return json.loads(data, object_pairs_hook=_object_pairs, parse_constant=_bad_constant)


def _keys(obj, expected, context):
    _require(isinstance(obj, dict), f"{context}: expected object")
    _require(set(obj) == set(expected),
             f"{context}: missing={sorted(set(expected) - set(obj))}, "
             f"unknown={sorted(set(obj) - set(expected))}")


def _text(value, context):
    _require(isinstance(value, str) and bool(value.strip()),
             f"{context}: expected nonempty string")
    return value


def _integer(value, context, minimum=0):
    _require(type(value) is int and value >= minimum,
             f"{context}: expected integer >= {minimum} (not bool)")


def _path(path, base):
    path = Path(path)
    return path if path.is_absolute() else base / path


def _schema(spec):
    _keys(spec, ("schema_version", "run_id", "metadata", "mosaic",
                 "banks", "code_files"), "spec")
    _require(type(spec["schema_version"]) is int and spec["schema_version"] == 1,
             "unsupported schema_version (expected integer 1)")
    _text(spec["run_id"], "run_id")
    meta = spec["metadata"]
    _keys(meta, ("path", "id_col", "patient_col", "label_col"), "metadata")
    for key in meta:
        _text(meta[key], f"metadata.{key}")
    _require(len({meta[k] for k in ("id_col", "patient_col", "label_col")}) == 3,
             "metadata identity columns must be distinct")
    mos = spec["mosaic"]
    _keys(mos, ("path", "filename_suffix", "level", "patch_size",
                "expected_slides", "expected_tiles"), "mosaic")
    for key in ("path", "filename_suffix"):
        _text(mos[key], f"mosaic.{key}")
    _require(not any(c in mos["filename_suffix"] for c in "/\\"),
             "filename_suffix must not contain a path separator")
    _integer(mos["level"], "mosaic.level")
    for key in ("patch_size", "expected_slides", "expected_tiles"):
        _integer(mos[key], f"mosaic.{key}", 1)
    _require(isinstance(spec["banks"], list) and bool(spec["banks"]),
             "banks must be a nonempty list")
    names = set()
    for bank in spec["banks"]:
        _keys(bank, ("name", "path", "feature_dim", "dtypes"), "bank")
        for key in ("name", "path"):
            _text(bank[key], f"bank.{key}")
        _require(bank["name"] not in names, f"duplicate bank name: {bank['name']}")
        names.add(bank["name"])
        _integer(bank["feature_dim"], f"{bank['name']}.feature_dim", 1)
        dtypes = bank["dtypes"]
        _require(isinstance(dtypes, list) and bool(dtypes)
                 and all(type(d) is str and d in ("float16", "float32") for d in dtypes)
                 and len(dtypes) == len(set(dtypes)),
                 f"{bank['name']}: dtypes must be unique supported float16/float32 names")
    _require(isinstance(spec["code_files"], list) and bool(spec["code_files"]),
             "code_files must be a nonempty list")
    names = set()
    for code in spec["code_files"]:
        _keys(code, ("name", "path"), "code_file")
        for key in ("name", "path"):
            _text(code[key], f"code_file.{key}")
        _require(code["name"] not in names, f"duplicate code_file name: {code['name']}")
        names.add(code["name"])


def _binding(path, kind, name, *, data=None):
    path = Path(path)
    if data is None:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    else:
        digest = hashlib.sha256(data)
        size = len(data)
    _require(path.is_file(), f"{kind} {name}: not a regular input file: {path}")
    stat = path.stat()
    return {
        "kind": kind, "name": name, "path": str(path),
        "resolved_path": str(path.resolve(strict=True)),
        "size": size, "sha256": digest.hexdigest(),
        "device": stat.st_dev, "inode": stat.st_ino,
    }


def _pickle(data, context):
    # Only trusted local legacy pickles: pickle is NOT safe for untrusted input.
    try:
        for opcode, _, position in pickletools.genops(io.BytesIO(data)):
            if opcode.name == "STOP":
                _require(position + 1 == len(data),
                         f"{context}: trailing bytes or additional serialized objects")
                break
        else:
            raise AuditError(f"{context}: pickle has no STOP opcode")
        return pickle.loads(data)
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(f"{context}: invalid pickle: {exc}") from exc


def _coordinates(tiles, context):
    _require(isinstance(tiles, list) and bool(tiles),
             f"{context}: expected nonempty ordered tile list")
    coords = []
    seen = set()
    for i, tile in enumerate(tiles):
        _require(isinstance(tile, dict) and "wsi_loc" in tile,
                 f"{context} tile {i}: missing wsi_loc dictionary")
        loc = tile["wsi_loc"]
        _require(isinstance(loc, list) and len(loc) == 2
                 and all(type(c) is int for c in loc),
                 f"{context} tile {i}: wsi_loc must be two integer list components")
        xy = tuple(loc)
        _require(xy not in seen, f"{context} tile {i}: duplicate physical coordinate {xy}")
        seen.add(xy)
        coords.append(xy)
    return coords


def _metadata(data, config):
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    headers = next(reader, None)
    _require(headers is not None and len(headers) == len(set(headers))
             and all(h.strip() for h in headers),
             "metadata: absent, blank or duplicate CSV headers")
    for col in ("id_col", "patient_col", "label_col"):
        _require(config[col] in headers, f"metadata: missing {col}={config[col]}")
    ids = set()
    patient_labels = defaultdict(set)
    for line, cells in enumerate(reader, 2):
        _require(len(cells) == len(headers),
                 f"metadata row {line}: expected {len(headers)} fields, got {len(cells)}")
        row = dict(zip(headers, cells))
        for col in ("id_col", "patient_col", "label_col"):
            _text(row[config[col]], f"metadata row {line} {config[col]}")
        sid = row[config["id_col"]]
        _require(sid not in ids, f"metadata row {line}: duplicate scan ID {sid!r}")
        ids.add(sid)
        patient_labels[row[config["patient_col"]]].add(row[config["label_col"]])
    return ids, headers, patient_labels


def _canonical(data, config):
    entries = _pickle(data, "canonical mosaic")
    _require(isinstance(entries, list), "canonical mosaic: expected list of (filename, mosaic)")
    coords = {}
    suffix = config["filename_suffix"]
    for i, entry in enumerate(entries):
        _require(isinstance(entry, tuple) and len(entry) == 2,
                 f"canonical mosaic entry {i}: expected (filename, mosaic) tuple")
        filename, tiles = entry
        _require(isinstance(filename, str) and filename.endswith(suffix)
                 and len(filename) > len(suffix) and Path(filename).name == filename,
                 f"canonical mosaic entry {i}: invalid filename/suffix {filename!r}")
        sid = filename[:-len(suffix)]
        _require(sid not in coords, f"canonical mosaic: duplicate scan ID {sid!r}")
        coords[sid] = _coordinates(tiles, f"canonical mosaic {sid}")
    _require(len(coords) == config["expected_slides"],
             f"canonical mosaic slide count {len(coords)} != {config['expected_slides']}")
    total = sum(map(len, coords.values()))
    _require(total == config["expected_tiles"],
             f"canonical mosaic tile count {total} != {config['expected_tiles']}")
    return coords


def _bank_inventory(root, expected, name):
    _require(root.is_dir(), f"bank {name}: not a directory: {root}")
    files = {p.stem: p for p in root.iterdir() if p.suffix == ".pkl"}
    _require(set(files) == expected,
             f"bank {name}: missing={sorted(expected - set(files))[:10]}, "
             f"extra={sorted(set(files) - expected)[:10]}")
    _require(len(files) == len(expected),
             f"bank {name}: duplicate pickle stems")
    for sid, path in files.items():
        _require(path.is_file(), f"bank {name}/{sid}: not a regular file")
    return files


def _inside(path, parent):
    return path == parent or parent in path.parents


def _output_path(out, out_root, paths, roots):
    target = out.resolve()
    base = out_root.resolve(strict=True)
    _require(target != base and _inside(target, base),
             f"output must be a new directory beneath PROSPER_OUT: {out}")
    for path in paths:
        resolved = path.resolve(strict=True)
        _require(not _inside(resolved, target) and not _inside(target, resolved),
                 f"output overlaps input: {path}")
    for root in roots:
        resolved = root.resolve(strict=True)
        _require(not _inside(resolved, target) and not _inside(target, resolved),
                 f"output overlaps bank: {root}")
    return target


def _claim_output(out, target):
    parent_fd = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        expected_parent = target.parent.stat()
        parent = os.fstat(parent_fd)
        _require((parent.st_dev, parent.st_ino)
                 == (expected_parent.st_dev, expected_parent.st_ino),
                 f"output parent retargeted before claim: {out.parent}")
        os.mkdir(out.name, dir_fd=parent_fd)
        observed = os.stat(out.name, dir_fd=parent_fd, follow_symlinks=False)
        directory_fd = os.open(out.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=parent_fd)
        try:
            claimed = os.fstat(directory_fd)
            current = os.stat(out.name, dir_fd=parent_fd, follow_symlinks=False)
            _require((observed.st_dev, observed.st_ino)
                     == (claimed.st_dev, claimed.st_ino)
                     == (current.st_dev, current.st_ino),
                     "output directory changed during acquisition")
            _require(not os.listdir(directory_fd),
                     "newly claimed output directory is not empty")
            return directory_fd
        except Exception:
            os.close(directory_fd)
            raise
    finally:
        os.close(parent_fd)


def _atomic_bytes(directory_fd, name, data):
    part = "." + name + ".part"
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o666, dir_fd=directory_fd)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)


def _read_output(directory_fd, name):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    with os.fdopen(fd, "rb") as handle:
        _require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode),
                 f"written {name}: not a regular file")
        return handle.read()


def _claimed_path(directory_fd, target):
    stat = os.fstat(directory_fd)
    try:
        current = target.stat()
    except FileNotFoundError as exc:
        raise AuditError(f"claimed output directory moved or removed: {target}") from exc
    _require((stat.st_dev, stat.st_ino) == (current.st_dev, current.st_ino),
             f"claimed output directory changed: {target}")


def _encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")


def _recheck(bindings, roots, inventories, expected):
    for name, root in roots.items():
        _require(str(root.resolve(strict=True)) == inventories[name]["resolved_root"],
                 f"bank {name}: root retargeted")
        current = _bank_inventory(root, expected, name)
        _require({sid: str(p.resolve(strict=True)) for sid, p in current.items()}
                 == inventories[name]["files"],
                 f"bank {name}: file inventory or symlink target changed")
    for original in bindings:
        path = Path(original["path"])
        current = _binding(path, original["kind"], original["name"])
        _require(current == original,
                 f"{original['kind']} {original['name']}: late input mutation or retarget: {path}")


def _verify_outputs(directory_fd, spec, coords, summary, expected_inventory):
    payload = _read_output(directory_fd, "inventory.csv")
    _require(payload == expected_inventory, "written inventory: differs from validated rows")
    rows = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    columns = ("bank", "slide_id", "canonical_tiles", "feature_rows", "dtype",
               "feature_path", "resolved_path", "size", "sha256")
    _require(rows.fieldnames == list(columns), "written inventory: invalid headers")
    expected = {(b["name"], sid) for b in spec["banks"] for sid in coords}
    seen = set()
    dtype_counts = defaultdict(Counter)
    bank_totals = Counter()
    count = 0
    for row in rows:
        _require(None not in row and None not in row.values(), "written inventory: malformed row")
        key = row["bank"], row["slide_id"]
        _require(key in expected and key not in seen,
                 f"written inventory: unexpected or duplicate key {key}")
        seen.add(key)
        sid = row["slide_id"]
        tiles = len(coords[sid])
        _require(int(row["canonical_tiles"]) == int(row["feature_rows"]) == tiles,
                 f"written inventory: wrong tile count {key}")
        b = next(b for b in spec["banks"] if b["name"] == row["bank"])
        _require(row["dtype"] in b["dtypes"] and len(row["sha256"]) == 64
                 and int(row["size"]) > 0 and bool(row["feature_path"])
                 and bool(row["resolved_path"]), f"written inventory: invalid binding {key}")
        dtype_counts[row["bank"]][row["dtype"]] += 1
        bank_totals[row["bank"]] += tiles
        count += 1
    _require(seen == expected and count == len(expected),
             f"written inventory: incomplete coverage ({len(seen)}/{len(expected)})")
    _require(summary["inventory_rows"] == count
             and summary["canonical_slides"] == len(coords)
             and summary["canonical_tiles"] == sum(map(len, coords.values()))
             and summary["bank_tile_totals"] == dict(bank_totals)
             and summary["bank_dtype_counts"] == {k: dict(v) for k, v in dtype_counts.items()},
             "written manifest: counts, coverage or dtypes do not reconcile")
    try:
        summary_data = _read_output(directory_fd, "summary.json")
        written_summary = _json(summary_data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"written summary.json: malformed output: {exc}") from exc
    _require(written_summary == summary, "written manifest: differs from validated manifest")
    _require(summary_data == _encoded(summary), "written manifest: byte content changed")
    hashes = {}
    for name, expected_data in (("inventory.csv", payload), ("summary.json", summary_data)):
        actual = _read_output(directory_fd, name)
        _require(actual == expected_data, f"written {name}: changed during readback")
        hashes[name] = {"size": len(actual), "sha256": hashlib.sha256(actual).hexdigest()}
    return hashes


def audit(spec_path, spec_sha256, out, out_root):
    """Audit trusted inputs; claim a new output directory only after preflight."""
    _require(isinstance(spec_sha256, str) and len(spec_sha256) == 64
             and all(c in "0123456789abcdefABCDEF" for c in spec_sha256),
             "spec-sha256 must be 64 hex characters")
    _require(out_root and str(out_root).strip(), "PROSPER_OUT must be explicitly nonempty")
    spec_path = _path(Path(spec_path), Path.cwd())
    spec_data = spec_path.read_bytes()
    _require(hashlib.sha256(spec_data).hexdigest() == spec_sha256.lower(),
             "spec SHA256 mismatch")
    spec = _json(spec_data)
    _schema(spec)
    base = spec_path.parent
    paths = [spec_path, _path(spec["metadata"]["path"], base),
             _path(spec["mosaic"]["path"], base)]
    code = [(_path(c["path"], base), c["name"]) for c in spec["code_files"]]
    code.append((_path(Path(__file__), Path.cwd()), "running_auditor"))
    paths.extend(p for p, _ in code)
    roots = {b["name"]: _path(b["path"], base) for b in spec["banks"]}
    resolved_roots = [r.resolve(strict=True) for r in roots.values()]
    _require(len(set(resolved_roots)) == len(roots), "duplicate/resolved bank roots")
    out = _path(Path(out), Path.cwd())
    out_root = _path(Path(out_root), Path.cwd())
    claimed_target = _output_path(out, out_root, paths, list(roots.values()))

    # Exclusive creation rejects competing runs; subsequent I/O uses only this handle.
    directory_fd = _claim_output(out, claimed_target)
    owned = False
    try:
        _claimed_path(directory_fd, claimed_target)
        _require(_output_path(out, out_root, paths, list(roots.values()))
                 == claimed_target, f"output lexical path retargeted: {out}")
        owned = True
        verification_started_at = _time()
        bindings = [_binding(spec_path, "spec", "spec", data=spec_data)]
        meta_data = paths[1].read_bytes()
        bindings.append(_binding(paths[1], "metadata", "metadata", data=meta_data))
        ids, headers, patient_labels = _metadata(meta_data, spec["metadata"])
        mosaic_data = paths[2].read_bytes()
        bindings.append(_binding(paths[2], "mosaic", "canonical", data=mosaic_data))
        coords = _canonical(mosaic_data, spec["mosaic"])
        del mosaic_data
        _require(ids == set(coords),
                 f"metadata vs mosaic: missing={sorted(ids - set(coords))[:10]}, "
                 f"extra={sorted(set(coords) - ids)[:10]}")
        for path, name in code:
            bindings.append(_binding(path, "code", name))
        aliases = []
        identities = {}
        for item in bindings:
            identity = (item["device"], item["inode"])
            if identity in identities:
                aliases.append([identities[identity], f"{item['kind']}:{item['name']}"])
            else:
                identities[identity] = f"{item['kind']}:{item['name']}"
        inventories = {}
        for name, root in roots.items():
            files = _bank_inventory(root, set(coords), name)
            inventories[name] = {
                "path": str(root), "resolved_root": str(root.resolve(strict=True)),
                "files": {sid: str(path.resolve(strict=True)) for sid, path in files.items()},
            }
        out_buffer = io.StringIO(newline="")
        writer = csv.writer(out_buffer)
        writer.writerow(("bank", "slide_id", "canonical_tiles", "feature_rows",
                         "dtype", "feature_path", "resolved_path", "size", "sha256"))
        dtype_counts = defaultdict(Counter)
        bank_totals = Counter()
        for bank in spec["banks"]:
            name = bank["name"]
            for sid in sorted(coords):
                path = roots[name] / f"{sid}.pkl"
                data = path.read_bytes()
                item = _binding(path, "feature", f"{name}/{sid}", data=data)
                identity = (item["device"], item["inode"])
                _require(identity not in identities,
                         f"file alias: {name}/{sid} aliases {identities.get(identity)}")
                identities[identity] = f"feature:{name}/{sid}"
                bindings.append(item)
                obj = _pickle(data, f"bank {name}/{sid}")
                del data
                _require(isinstance(obj, tuple) and len(obj) == 2,
                         f"bank {name}/{sid}: expected (features, mosaic) tuple")
                features, tiles = obj
                _require(isinstance(features, np.ndarray) and features.ndim == 2
                         and features.shape[1] == bank["feature_dim"]
                         and features.shape[0] == len(coords[sid]),
                         f"bank {name}/{sid}: wrong feature array shape/width/rows")
                dtype = features.dtype.name
                _require(dtype in bank["dtypes"] and features.dtype.kind == "f"
                         and features.dtype.isnative,
                         f"bank {name}/{sid}: unsupported feature dtype {features.dtype}")
                _require(bool(np.isfinite(features).all()),
                         f"bank {name}/{sid}: NaN/Inf in features")
                _require(_coordinates(tiles, f"bank {name}/{sid}") == coords[sid],
                         f"bank {name}/{sid}: ordered coordinate mismatch")
                del obj, features, tiles
                writer.writerow((name, sid, len(coords[sid]), len(coords[sid]), dtype,
                                 item["path"], item["resolved_path"],
                                 item["size"], item["sha256"]))
                dtype_counts[name][dtype] += 1
                bank_totals[name] += len(coords[sid])
        summary = {
            "schema_version": 1, "run_id": spec["run_id"],
            "scope": "feature-bank binding only; encoder identities declared/unverified",
            "canonical_slides": len(coords),
            "canonical_tiles": sum(map(len, coords.values())),
            "inventory_rows": len(coords) * len(roots),
            "bank_tile_totals": dict(bank_totals),
            "bank_dtype_counts": {k: dict(v) for k, v in dtype_counts.items()},
            "metadata_columns": headers, "patient_count": len(patient_labels),
            "multi_label_patient_count": sum(len(v) > 1 for v in patient_labels.values()),
            "declarations": spec, "input_bindings": bindings,
            "bank_roots": {k: {"path": str(r), "resolved_path": str(r.resolve(strict=True))}
                           for k, r in roots.items()},
            "declared_file_aliases": aliases,
            "environment": {
                "python": sys.version, "executable": sys.executable,
                "numpy": np.__version__, "pandas": pd.__version__,
                "hostname": socket.gethostname(), "platform": platform.platform(),
                "lsf": {k: os.environ[k] for k in ("LSB_JOBID", "LSB_JOBINDEX",
                         "LSB_QUEUE", "LSB_HOSTS") if os.environ.get(k)},
            },
            "verification_started_at": verification_started_at,
        }
        expected_inventory = out_buffer.getvalue().encode("utf-8")
        _atomic_bytes(directory_fd, "inventory.csv", expected_inventory)
        _atomic_bytes(directory_fd, "summary.json", _encoded(summary))
        _recheck(bindings, roots, inventories, set(coords))
        _require(_output_path(out, out_root, paths, list(roots.values()))
                 == claimed_target, f"output lexical path retargeted: {out}")
        _claimed_path(directory_fd, claimed_target)
        hashes = _verify_outputs(directory_fd, spec, coords, summary, expected_inventory)
        _require(_output_path(out, out_root, paths, list(roots.values()))
                 == claimed_target, f"output lexical path retargeted: {out}")
        _claimed_path(directory_fd, claimed_target)
        receipt = {
            "schema_version": 1, "run_id": spec["run_id"],
            "scope": "feature-bank binding only",
            "outputs": hashes,
            "initial_verification_at": summary["verification_started_at"],
            "final_verification_at": _time(),
            "limitations": [
                "Declared encoder identities are unverified; matching widths cannot detect swapped banks.",
                "No proof of historical extraction, checkpoint identity, loaded bytecode, matrices or scientific acceptance.",
                "Two passes are not an atomic snapshot; freeze inputs and enforce recorded hashes downstream.",
                "Competing creation is refused, but an active run directory or its physical parents must not be renamed or replaced; an empty substitution before first observation is not detectable.",
                "Pickle input must be trusted locally; untrusted pickle ingestion is unsafe.",
            ],
        }
        _atomic_bytes(directory_fd, "COMPLETE.json", _encoded(receipt))
        return receipt
    except Exception as exc:
        if owned:
            try:
                _atomic_bytes(directory_fd, "FAILED.json", _encoded({
                    "failed_at": _time(), "type": type(exc).__name__, "error": str(exc),
                }))
            except Exception:
                pass
        raise
    finally:
        os.close(directory_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        audit(args.spec, args.spec_sha256, args.out, os.environ.get("PROSPER_OUT"))
    except Exception as exc:
        print(f"feature-bank audit FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"feature-bank audit COMPLETE: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
