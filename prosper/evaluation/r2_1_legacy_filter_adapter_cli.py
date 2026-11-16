"""R2.1 filtering route (a): adapt a historical UNI2/Virchow2 tumour-only distance matrix.

Specification: [decision record] "B(a) -- specification review 001 disposition and candidate 002" plus the
focused re-check amendments W-1..W-5. One invocation handles one (encoder, method) pair:
it reads the frozen tumour-only ``_no_same_patient`` pickle, the accepted retrieval
population and the legacy ``tumor_patch_indices_dict.pkl`` of the same encoder and
writes, into an ABSENT output directory, the inputs that the frozen evaluator
``r2_1_filter_eval_cli.py`` (``main`` path) expects for a filtered arm:

* ``_filter_summary.csv`` -- every population slide once, population order,
  ``written`` in {0, 1};
* ``matrix_written.pkl`` -- the written slides x written slides sub-matrix, population
  order, float64, values copied exactly (no symmetrisation or transposition);
* ``nonreturn_set.json`` -- sorted ids of the non-returned set N;
* ``ADAPTER_META.json`` -- provenance and per-slide diagnostics.

Written rule: from cross-patient population cells only. N is the set of population slides
whose cross-patient row AND column cells are all NaN; those slides get ``written = 0``,
every other slide ``written = 1``. The diagonal and the same-patient cells never decide
``written``. Every check completes before the output directory is created; on any failure
nothing is created and the process exits non-zero with a message on stderr.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import os
import pickle
import platform
import re
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SENTINEL = 1e6
LOW_TILE_THRESHOLD = 8
REASON_NONRETURN = "legacy_zero_kept_tiles"
SUMMARY_NAME = "_filter_summary.csv"
MATRIX_NAME = "matrix_written.pkl"
NONRETURN_NAME = "nonreturn_set.json"
META_NAME = "ADAPTER_META.json"
SUMMARY_HEADER = ("slide_index", "slide_id", "patient_id", "diagnostic_group", "written",
                  "reason")
POPULATION_REQUIRED = ("query_index", "reference_index", "slide_id", "patient_id",
                       "diagnostic_group")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_INT = re.compile(r"^(0|[1-9][0-9]*)$")
_MAX_LISTED = 20


class AdapterError(RuntimeError):
    """An input violates the route-(a) adapter specification; nothing is written."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_input(path: Path, label: str, expected: Optional[str]) -> Dict[str, object]:
    """Read the whole file once; hash exactly the bytes that are parsed."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise AdapterError(f"{label}: cannot read {path}: {exc}") from exc
    digest = sha256_bytes(data)
    if expected is not None and digest != expected:
        raise AdapterError(f"{label}: sha256 {digest} != expected {expected} ({path})")
    return {"path": str(path), "resolved_path": str(Path(os.path.abspath(path))),
            "sha256": digest, "size_bytes": len(data), "data": data}


def _listed(ids: Sequence[str]) -> str:
    ids = list(ids)
    more = "" if len(ids) <= _MAX_LISTED else f" ... (+{len(ids) - _MAX_LISTED} more)"
    return ", ".join(ids[:_MAX_LISTED]) + more


def parse_population(data: bytes, label: str = "population") -> pd.DataFrame:
    try:
        pop = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, na_filter=False,
                          float_precision="round_trip")
    except Exception as exc:  # noqa: BLE001 - any parse failure is a specification failure
        raise AdapterError(f"{label}: cannot parse CSV: {exc}") from exc
    missing = [c for c in POPULATION_REQUIRED if c not in pop.columns]
    if missing:
        raise AdapterError(f"{label}: lacks columns {missing}")
    if list(pop.columns).count("slide_id") != 1 or pop.columns.duplicated().any():
        raise AdapterError(f"{label}: duplicated column names")
    if len(pop) == 0:
        raise AdapterError(f"{label}: no rows")
    for c in POPULATION_REQUIRED:
        bad = [str(v) for v in pop[c] if not isinstance(v, str) or v.strip() == ""
               or v != v.strip()]
        if bad:
            raise AdapterError(f"{label}: empty or whitespace-padded {c} values")
    for c in ("query_index", "reference_index"):
        if not all(_CANONICAL_INT.match(v) for v in pop[c]):
            raise AdapterError(f"{label}: {c} must be canonical non-negative integers")
        if [int(v) for v in pop[c]] != list(range(len(pop))):
            raise AdapterError(f"{label}: {c} must equal the 0-based row order 0..n-1")
    if pop.slide_id.duplicated().any():
        dup = sorted(set(pop.slide_id[pop.slide_id.duplicated()]))
        raise AdapterError(f"{label}: duplicate slide_id {_listed(dup)}")
    return pop.reset_index(drop=True)


def parse_matrix(data: bytes, label: str = "matrix") -> pd.DataFrame:
    try:
        df = pickle.loads(data)
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label}: cannot unpickle: {exc}") from exc
    if not isinstance(df, pd.DataFrame):
        raise AdapterError(f"{label}: not a pandas DataFrame ({type(df).__name__})")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise AdapterError(f"{label}: empty matrix")
    if not all(np.dtype(t) == np.float64 for t in df.dtypes):
        raise AdapterError(f"{label}: every column must be float64 "
                           f"(found {sorted({str(t) for t in df.dtypes})})")
    return df


def _axis_ids(axis, what: str, label: str) -> List[str]:
    ids = [str(v) for v in axis]
    if len(set(ids)) != len(ids):
        seen, dup = set(), set()
        for v in ids:
            (dup if v in seen else seen).add(v)
        raise AdapterError(f"{label}: duplicated {what} id after str-cast: {_listed(sorted(dup))}")
    return ids


def parse_index_dict(data: bytes, label: str = "index-dict") -> Dict[str, int]:
    """Kept-tile count per str-cast key (W-1)."""
    try:
        obj = pickle.loads(data)
    except Exception as exc:  # noqa: BLE001
        raise AdapterError(f"{label}: cannot unpickle: {exc}") from exc
    if not isinstance(obj, dict):
        raise AdapterError(f"{label}: not a dict ({type(obj).__name__})")
    counts: Dict[str, int] = {}
    for key, value in obj.items():
        k = str(key)
        if k in counts:
            raise AdapterError(f"{label}: duplicate key after str-cast: {k}")
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind not in "iu":
            desc = (f"ndarray ndim={value.ndim} dtype={value.dtype}"
                    if isinstance(value, np.ndarray) else type(value).__name__)
            raise AdapterError(f"{label}: value for key {k} is not a 1-D integer array ({desc})")
        counts[k] = int(value.shape[0])
    return counts


def analyse(pop: pd.DataFrame, df: pd.DataFrame, counts: Dict[str, int],
            label: str = "matrix") -> Dict[str, object]:
    """Run every specification check and derive N; raise AdapterError on any violation."""
    rows = _axis_ids(df.index, "row", label)
    cols = _axis_ids(df.columns, "column", label)
    if rows != cols:
        raise AdapterError(f"{label}: index != columns after str-cast")
    ids = pop.slide_id.tolist()
    pos = {s: i for i, s in enumerate(rows)}
    absent = [s for s in ids if s not in pos]
    if absent:
        raise AdapterError(f"{label}: population slides missing from the axes: {_listed(absent)}")
    missing_dict = [s for s in ids if s not in counts]
    if missing_dict:
        raise AdapterError(f"index-dict: population slides missing: {_listed(missing_dict)}")

    sel = np.array([pos[s] for s in ids], dtype=np.int64)
    values = df.to_numpy(dtype=np.float64, copy=False)
    sub = values[np.ix_(sel, sel)]
    n = len(ids)
    patients = pop.patient_id.to_numpy(dtype=object)
    same = patients[:, None] == patients[None, :]
    eye = np.eye(n, dtype=bool)
    same_off = same & ~eye
    cross = ~same

    diag = np.diagonal(sub).copy()
    bad_diag = ~(np.isnan(diag) | (diag == SENTINEL))
    if bad_diag.any():
        raise AdapterError(f"{label}: diagonal not NaN/1e6 for {_listed([ids[i] for i in np.flatnonzero(bad_diag)])}")
    bad_same = same_off & (sub != SENTINEL)  # NaN != 1e6 is True, so NaN is also rejected
    if bad_same.any():
        r, c = np.nonzero(bad_same)
        raise AdapterError(f"{label}: {r.size} same-patient population cells != 1e6, e.g. "
                           f"({ids[r[0]]}, {ids[c[0]]})")
    no_partner = ~cross.any(axis=1)
    if no_partner.any():
        raise AdapterError(f"population slides without any cross-patient slide: "
                           f"{_listed([ids[i] for i in np.flatnonzero(no_partner)])}")
    used_sent = cross & (sub == SENTINEL)
    if used_sent.any():
        r, c = np.nonzero(used_sent)
        raise AdapterError(f"{label}: {r.size} cross-patient cells == 1e6, e.g. ({ids[r[0]]}, {ids[c[0]]})")
    infs = cross & np.isinf(sub)
    if infs.any():
        r, c = np.nonzero(infs)
        raise AdapterError(f"{label}: {r.size} cross-patient cells are +-inf, e.g. ({ids[r[0]]}, {ids[c[0]]})")

    nan = np.isnan(sub)
    row_cross_nan = (nan & cross).sum(axis=1)
    col_cross_nan = (nan & cross).sum(axis=0)
    n_cross = cross.sum(axis=1)
    row_all = row_cross_nan == n_cross
    col_all = col_cross_nan == n_cross
    in_n = row_all & col_all
    written = ~in_n
    if not written.any():
        raise AdapterError(f"{label}: no population slide would be written")

    ww = cross & written[:, None] & written[None, :]
    partial = ww & nan
    if partial.any():
        r, c = np.nonzero(partial)
        bad = sorted({ids[i] for i in r} | {ids[j] for j in c})
        raise AdapterError(f"{label}: {r.size} NaN cross-patient cells between written slides "
                           f"(partial-NaN reject), slides {_listed(bad)}")
    wn = cross & ((written[:, None] & in_n[None, :]) | (in_n[:, None] & written[None, :]))
    finite_to_n = wn & ~nan
    if finite_to_n.any():
        r, c = np.nonzero(finite_to_n)
        raise AdapterError(f"{label}: finite cross-patient cell towards a non-returned slide, "
                           f"e.g. ({ids[r[0]]}, {ids[c[0]]})")

    kept = np.array([counts[s] for s in ids], dtype=np.int64)
    disagree = (kept == 0) != in_n
    if disagree.any():
        bad = [f"{ids[i]}(count={int(kept[i])}, in_N={bool(in_n[i])})"
               for i in np.flatnonzero(disagree)]
        raise AdapterError(f"index-dict disagreement (count==0 <=> non-returned): {_listed(bad)}")

    per_slide = []
    for i, s in enumerate(ids):
        d = diag[i]
        per_slide.append({
            "slide_id": s,
            "written": int(written[i]),
            "diagonal": "NaN" if np.isnan(d) else float(d),
            "same_patient_partners": int(same_off[i].sum()),
            "same_patient_cells_checked": int(2 * same_off[i].sum()),
            "same_patient_cells_all_1e6": True,
            "cross_patient_partners": int(n_cross[i]),
            "cross_patient_row_nan": int(row_cross_nan[i]),
            "cross_patient_col_nan": int(col_cross_nan[i]),
            "kept_tiles": int(kept[i]),
        })
    return {"ids": ids, "sel": sel, "written": written, "in_n": in_n, "kept": kept,
            "diag": diag, "per_slide": per_slide, "values": values,
            "axis_len": len(rows), "extra_axis_ids": len(set(rows) - set(ids))}


def build_outputs(pop: pd.DataFrame, res: Dict[str, object]) -> Dict[str, bytes]:
    """Serialize summary, written matrix and N in memory (no filesystem writes)."""
    ids: List[str] = res["ids"]
    written = res["written"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SUMMARY_HEADER)
    for i, row in enumerate(pop.itertuples(index=False)):
        wr = int(written[i])
        w.writerow([row.query_index, row.slide_id, row.patient_id, row.diagnostic_group, wr,
                    "" if wr else REASON_NONRETURN])
    summary = buf.getvalue().encode("utf-8")

    keep = np.flatnonzero(written)
    sel = res["sel"][keep]
    kept_ids = [ids[i] for i in keep]
    block = np.array(res["values"][np.ix_(sel, sel)], dtype=np.float64, copy=True)
    out_df = pd.DataFrame(block, index=pd.Index(kept_ids, dtype=object),
                          columns=pd.Index(kept_ids, dtype=object))
    mbuf = io.BytesIO()
    out_df.to_pickle(mbuf, compression=None)
    matrix = mbuf.getvalue()
    # Self-check: the serialized frame must round-trip to the exact selected values.
    back = pickle.loads(matrix)
    if (list(map(str, back.index)) != kept_ids or list(map(str, back.columns)) != kept_ids
            or not np.array_equal(back.to_numpy(), block, equal_nan=True)
            or not all(np.dtype(t) == np.float64 for t in back.dtypes)):
        raise AdapterError("internal: written matrix failed its round-trip self-check")
    patients = pop.patient_id.to_numpy(dtype=object)[keep]
    cross = patients[:, None] != patients[None, :]
    if np.isnan(block[cross]).any():
        raise AdapterError("internal: NaN on a cross-patient cell of the written matrix")

    nonret = sorted(ids[i] for i in np.flatnonzero(res["in_n"]))
    nonreturn = (json.dumps(nonret, indent=1) + "\n").encode("utf-8")
    return {SUMMARY_NAME: summary, MATRIX_NAME: matrix, NONRETURN_NAME: nonreturn}


def _code_sha256() -> str:
    with open(__file__, "rb") as fh:
        return sha256_bytes(fh.read())


def build_meta(argv: List[str], inputs: Dict[str, Dict[str, object]], pop: pd.DataFrame,
               res: Dict[str, object], outputs: Dict[str, bytes]) -> bytes:
    ids: List[str] = res["ids"]
    kept = res["kept"]
    in_n = res["in_n"]
    written_ids = [ids[i] for i in np.flatnonzero(res["written"])]
    nonret_ids = sorted(ids[i] for i in np.flatnonzero(in_n))
    meta = {
        "tool": "r2_1_legacy_filter_adapter_cli",
        "contract": "ADR0021 B(a) candidate 002 + RECHECK_002 W-1..W-5",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "sys_argv": list(sys.argv),
        "argv": list(argv),
        "lsf_job": os.environ.get("LSB_JOBID"),
        "code": {"path": str(Path(os.path.abspath(__file__))), "sha256": _code_sha256()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "python": platform.python_version()},
        "inputs": {k: {kk: vv for kk, vv in v.items() if kk != "data"} for k, v in inputs.items()},
        "population_slides": len(ids),
        "population_patients": int(pop.patient_id.nunique()),
        "matrix_axis_length": res["axis_len"],
        "matrix_extra_axis_ids_outside_population": res["extra_axis_ids"],
        "written_count": len(written_ids),
        "written_ids": written_ids,
        "nonreturned_count": len(nonret_ids),
        "nonreturned_ids": nonret_ids,
        "nonreturn_reason": REASON_NONRETURN,
        "diagonal_counts": {"nan": int(np.isnan(res["diag"]).sum()),
                            "sentinel_1e6": int((res["diag"] == SENTINEL).sum())},
        "kept_tiles": {
            "min": int(kept.min()),
            "median": float(np.median(kept)),
            "low_threshold": LOW_TILE_THRESHOLD,
            "ids_below_threshold": [ids[i] for i in np.flatnonzero(kept < LOW_TILE_THRESHOLD)],
        },
        "per_slide": res["per_slide"],
        "outputs": {name: {"sha256": sha256_bytes(b), "size_bytes": len(b)}
                    for name, b in outputs.items()},
        "rules": {
            "written": "cross-patient population cells only; N = slides whose cross-patient row "
                       "AND column cells are all NaN; written=0 for N, 1 otherwise",
            "diagonal": "NaN or 1e6 required; diagnostic only",
            "same_patient": "all population same-patient off-diagonal cells must equal 1e6",
            "matrix_written": "written x written, population order, float64, exact copy, "
                              "no symmetrisation/transposition",
        },
    }
    return (json.dumps(meta, indent=1, allow_nan=False) + "\n").encode("utf-8")


def _write_all(out_dir: Path, files: Dict[str, bytes]) -> None:
    os.mkdir(out_dir)  # raises if it appeared since the pre-check
    created: List[Path] = []
    try:
        for name, data in files.items():
            p = out_dir / name
            with open(p, "xb") as fh:
                created.append(p)
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
    except BaseException:
        for p in created:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            out_dir.rmdir()
        except OSError:
            pass
        raise


def _sha_arg(value: str) -> str:
    v = value.strip().lower()
    if not _HEX64.match(v):
        raise argparse.ArgumentTypeError(f"not a 64-hex sha256: {value!r}")
    return v


def run(argv: List[str]) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True,
                   help="historical tumour-only _no_same_patient DataFrame pickle")
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--index-dict", type=Path, required=True,
                   help="legacy tumor_patch_indices_dict.pkl of the same encoder")
    p.add_argument("--out-dir", type=Path, required=True, help="must not exist")
    p.add_argument("--expected-matrix-sha256", type=_sha_arg)
    p.add_argument("--expected-population-sha256", type=_sha_arg)
    p.add_argument("--expected-index-dict-sha256", type=_sha_arg)
    a = p.parse_args(argv)

    out_dir: Path = a.out_dir
    if os.path.lexists(out_dir):
        raise AdapterError(f"output directory already exists: {out_dir}")
    parent = Path(os.path.abspath(out_dir)).parent
    if not parent.is_dir():
        raise AdapterError(f"output parent directory does not exist: {parent}")

    inputs = {
        "matrix": _read_input(a.matrix, "matrix", a.expected_matrix_sha256),
        "population": _read_input(a.population, "population", a.expected_population_sha256),
        "index_dict": _read_input(a.index_dict, "index-dict", a.expected_index_dict_sha256),
    }
    pop = parse_population(inputs["population"]["data"])
    df = parse_matrix(inputs["matrix"]["data"])
    counts = parse_index_dict(inputs["index_dict"]["data"])
    res = analyse(pop, df, counts)
    outputs = build_outputs(pop, res)
    meta = build_meta(argv, inputs, pop, res, outputs)
    if os.path.lexists(out_dir):
        raise AdapterError(f"output directory appeared during checks: {out_dir}")
    _write_all(out_dir, {**outputs, META_NAME: meta})
    return out_dir


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        out = run(args)
    except AdapterError as exc:
        print(f"r2_1_legacy_filter_adapter_cli: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out_dir": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
