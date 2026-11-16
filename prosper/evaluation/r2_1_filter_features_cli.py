"""R2.1 F1a: write H-Optimus tumour-filtered mosaic feature pickles.

For every slide of the accepted mosaic population (``slide_offsets.csv``),
load its H-Optimus feature pickle, verify it against the accepted mosaic
bundle (file sha256 and every tile coordinate), select the accepted mask of
the head whose held-out fold equals the slide's patient fold for the requested
label budget, and write ``(features[keep], [coords[i] for i in keep])`` --
the ``(features, coords)`` layout ``prosper.embeddings.similarity_cli`` reads.

Slides with fewer than ``--min-tiles`` retained tiles (historical ST2.0
``--min-tiles 8``) are not written: as queries they are non-returns, as
references they are absent from that arm's library.

Outputs are never overwritten. An existing output is accepted only if its
bytes equal the bytes this run would write; otherwise the run fails.

``--passthrough`` writes all-True-mask copies of the listed slides through the
same code path into ``<out-root>/passthrough/`` (fidelity check).

``--head-fold F`` ([decision record] section B, query-fold design): filter EVERY population
slide with the mask of head ``F`` (the head whose held-out fold is F), not the
slide's own fold, and write the bank to ``<out-root>/budget{b}/fold{F}/``
(default out-root ``r2_1_filter_20261003/features_qf``). That bank is both the
filtered query source for fold-F queries and fold F's filtered reference library.
Retained counts are checked against ``mask_counts.csv.gz`` rows with
``held_out_fold == F``; the summary records ``head_fold == F`` for every slide plus
its own ``slide_fold`` and ``own_fold_query``. Without ``--head-fold`` the
behaviour (own-fold head, ``<out-root>/budget{b}/``) is unchanged.
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
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PICKLE_PROTOCOL = 4  # protocol of the source H-Optimus pickles (b"\x80\x04")
FEATURE_WIDTH = 1536
SUMMARY_HEADER = ("slide_index", "slide_id", "patient_id", "diagnostic_group",
                  "slide_fold", "head_fold", "budget_percent", "original_tiles",
                  "retained_tiles", "written", "reason", "output_bytes", "sha256")
SUMMARY_HEADER_QF = SUMMARY_HEADER + ("own_fold_query",)


class FilterError(RuntimeError):
    """An input disagrees with the accepted mosaic bundle."""


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _default_out() -> Path:
    return Path(os.environ.get("PROSPER_OUT", "")) if os.environ.get("PROSPER_OUT") else None


def load_offsets(mosaic_dir: Path) -> pd.DataFrame:
    off = pd.read_csv(mosaic_dir / "slide_offsets.csv",
                      dtype={"slide_id": str, "patient_id": str, "feature_sha256": str})
    need = ["slide_index", "slide_id", "patient_id", "diagnostic_group", "slide_fold",
            "tile_start", "tile_stop", "feature_rows", "feature_sha256"]
    missing = [c for c in need if c not in off.columns]
    if missing:
        raise FilterError(f"slide_offsets.csv lacks columns {missing}")
    if not (off.slide_index.to_numpy() == np.arange(len(off))).all():
        raise FilterError("slide_offsets slide_index is not 0..n-1")
    if not (off.tile_stop - off.tile_start == off.feature_rows).all():
        raise FilterError("slide_offsets tile ranges disagree with feature_rows")
    if not (off.tile_start.iloc[1:].to_numpy() == off.tile_stop.iloc[:-1].to_numpy()).all():
        raise FilterError("slide_offsets tile ranges are not contiguous")
    if off.slide_id.duplicated().any():
        raise FilterError("duplicate slide_id in slide_offsets")
    return off


def load_masks(mosaic_dir: Path, budget: int, folds: Iterable[int]) -> Dict[int, np.ndarray]:
    out = {}
    for f in sorted(set(int(x) for x in folds)):
        m = np.load(mosaic_dir / f"fold{f}_budget{budget}_mask.npy", mmap_mode="r")
        p = np.load(mosaic_dir / f"fold{f}_budget{budget}_predictions.npy", mmap_mode="r")
        if m.dtype != np.bool_ or m.ndim != 1 or p.shape[0] != m.shape[0]:
            raise FilterError(f"fold{f}_budget{budget} mask/prediction shape or dtype mismatch")
        out[f] = (np.asarray(m), np.asarray(p))
    return out


def load_feature_pickle(path: Path) -> Tuple[np.ndarray, list, bytes]:
    data = path.read_bytes()
    obj = pickle.loads(data)
    if not (isinstance(obj, tuple) and len(obj) == 2):
        raise FilterError(f"{path.name}: not a (features, coords) 2-tuple")
    feats, coords = obj
    if not isinstance(feats, np.ndarray) or feats.ndim != 2:
        raise FilterError(f"{path.name}: features are not a 2-D ndarray")
    if not isinstance(coords, list) or len(coords) != feats.shape[0]:
        raise FilterError(f"{path.name}: coords length {len(coords) if isinstance(coords, list) else '?'}"
                          f" != feature rows {feats.shape[0]}")
    return feats, coords, data


def verify_slide(row, feats: np.ndarray, coords: list, data: bytes,
                 coordinates: np.ndarray, *, width: int = FEATURE_WIDTH) -> None:
    sid = row.slide_id
    digest = hashlib.sha256(data).hexdigest()
    if digest != row.feature_sha256:
        raise FilterError(f"{sid}: feature sha256 {digest} != accepted {row.feature_sha256}")
    n = int(row.feature_rows)
    if feats.shape != (n, width):
        raise FilterError(f"{sid}: feature shape {feats.shape} != ({n}, {width})")
    try:
        locs = np.asarray([c["wsi_loc"] for c in coords], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise FilterError(f"{sid}: malformed coords ({exc})") from exc
    ref = coordinates[int(row.tile_start):int(row.tile_stop)]
    if locs.shape != ref.shape or not np.array_equal(locs, ref):
        bad = int(np.flatnonzero((locs != ref).any(axis=1))[0]) if locs.shape == ref.shape else -1
        raise FilterError(f"{sid}: coordinates disagree with coordinates.npy (first bad row {bad})")


def filtered_bytes(feats: np.ndarray, coords: list, keep: np.ndarray) -> bytes:
    idx = np.flatnonzero(keep)
    obj = (feats[idx], [coords[i] for i in idx])
    return pickle.dumps(obj, protocol=PICKLE_PROTOCOL)


def write_once(path: Path, data: bytes) -> str:
    """Write ``data`` to ``path`` unless an identical file exists; never overwrite."""
    if path.exists():
        if sha256_file(path) != hashlib.sha256(data).hexdigest():
            raise FilterError(f"{path} exists with different bytes; refusing to overwrite")
        return "existing_identical"
    tmp = path.parent / f".{path.name}.{os.getpid()}.partial"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    if path.exists():
        tmp.unlink()
        raise FilterError(f"{path} appeared during write; refusing to overwrite")
    os.rename(tmp, path)
    return "written"


def write_record(path: Path, data: bytes) -> Path:
    """Records carry timestamps: a rerun adds a new numbered record, never replaces one."""
    target, n = path, 1
    while target.exists():
        n += 1
        target = path.with_name(f"{path.stem}.run{n}{path.suffix}")
    write_once(target, data)
    return target


def _write_csv(path: Path, header: Sequence[str], rows: List[Sequence]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    data = buf.getvalue().encode()
    write_once(path, data)
    return hashlib.sha256(data).hexdigest()


def run(*, mosaic_dir: Path, features_dir: Path, out_root: Path, budgets: Sequence[int],
        min_tiles: int, lo: int, hi: int, passthrough: Sequence[str],
        width: int = FEATURE_WIDTH, code_paths: Sequence[Path] = (),
        head_fold: Optional[int] = None) -> Dict:
    t0 = time.time()
    off = load_offsets(mosaic_dir)
    coordinates = np.load(mosaic_dir / "coordinates.npy", mmap_mode="r")
    if coordinates.shape != (int(off.tile_stop.iloc[-1]), 2):
        raise FilterError("coordinates.npy length disagrees with slide_offsets")
    counts = pd.read_csv(mosaic_dir / "mask_counts.csv.gz", dtype={"slide_id": str})
    qf = head_fold is not None
    if qf:
        head_fold = int(head_fold)
        if head_fold not in set(int(x) for x in off.slide_fold.unique()):
            raise FilterError(f"head fold {head_fold} is not a slide fold of the population")
        counts_hf = counts[counts.held_out_fold == head_fold].set_index(["budget_percent", "slide_id"])
        if counts_hf.index.duplicated().any():
            raise FilterError(f"duplicate mask_counts rows for head fold {head_fold}")
    counts = counts[counts.query_fold_eligible == 1].set_index(["budget_percent", "slide_id"])
    folds = [head_fold] if qf else off.slide_fold.unique()
    masks = {b: load_masks(mosaic_dir, b, folds) for b in budgets}
    sel = off.iloc[lo - 1:(hi or len(off))]
    full_range = lo == 1 and (hi == 0 or hi >= len(off))
    pt = set(passthrough)
    unknown = pt - set(off.slide_id)
    if unknown:
        raise FilterError(f"passthrough ids not in slide_offsets: {sorted(unknown)}")
    out_dirs = {b: out_root / f"budget{b}" / f"fold{head_fold}" if qf else out_root / f"budget{b}"
                for b in budgets}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    if pt:
        (out_root / "passthrough").mkdir(parents=True, exist_ok=True)
    rows = {b: [] for b in budgets}
    pt_rows = []
    read_bytes = 0
    for row in sel.itertuples(index=False):
        if not budgets and row.slide_id not in pt:
            continue
        src = features_dir / f"{row.slide_id}.pkl"
        feats, coords, data = load_feature_pickle(src)
        read_bytes += len(data)
        verify_slide(row, feats, coords, data, coordinates, width=width)
        a, z = int(row.tile_start), int(row.tile_stop)
        for b in budgets:
            mask, pred = masks[b][head_fold if qf else int(row.slide_fold)]
            keep = np.asarray(mask[a:z], dtype=bool)
            if not np.array_equal(keep, np.asarray(pred[a:z]) == 1):
                raise FilterError(f"{row.slide_id}: mask != (prediction == 1) for budget {b}")
            n_keep = int(keep.sum())
            if qf:
                c = counts_hf.loc[(b, row.slide_id)]
                if (int(c.retained_tiles) != n_keep or int(c.held_out_fold) != head_fold
                        or int(c.slide_fold) != int(row.slide_fold)
                        or int(c.query_fold_eligible) != int(head_fold == int(row.slide_fold))):
                    raise FilterError(f"{row.slide_id}: retained {n_keep} != mask_counts "
                                      f"{int(c.retained_tiles)} (or fold fields disagree) for "
                                      f"budget {b}, head fold {head_fold}")
            else:
                c = counts.loc[(b, row.slide_id)]
                if int(c.retained_tiles) != n_keep or int(c.held_out_fold) != int(row.slide_fold):
                    raise FilterError(f"{row.slide_id}: retained {n_keep} != mask_counts "
                                      f"{int(c.retained_tiles)} for budget {b}")
            if n_keep >= min_tiles:
                payload = filtered_bytes(feats, coords, keep)
                write_once(out_dirs[b] / f"{row.slide_id}.pkl", payload)
                rec = (1, "", len(payload), hashlib.sha256(payload).hexdigest())
            else:
                rec = (0, f"retained_tiles<{min_tiles}", "", "")
            if qf:
                rows[b].append((row.slide_index, row.slide_id, row.patient_id, row.diagnostic_group,
                                row.slide_fold, head_fold, b, int(row.feature_rows), n_keep, *rec,
                                int(int(row.slide_fold) == head_fold)))
            else:
                rows[b].append((row.slide_index, row.slide_id, row.patient_id, row.diagnostic_group,
                                row.slide_fold, row.slide_fold, b, int(row.feature_rows), n_keep, *rec))
        if row.slide_id in pt:
            payload = filtered_bytes(feats, coords, np.ones(feats.shape[0], dtype=bool))
            write_once(out_root / "passthrough" / f"{row.slide_id}.pkl", payload)
            pt_rows.append(dict(slide_id=row.slide_id, source=str(src),
                                source_sha256=row.feature_sha256, source_bytes=len(data),
                                output_sha256=hashlib.sha256(payload).hexdigest(),
                                output_bytes=len(payload),
                                byte_identical=payload == data))
    record = {
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
        "argv": sys.argv, "range": [lo, hi], "full_range": full_range,
        "min_tiles": min_tiles, "pickle_protocol": PICKLE_PROTOCOL,
        "inputs": {
            "mosaic_dir": str(mosaic_dir), "features_dir": str(features_dir),
            **{f"sha256:{n}": sha256_file(mosaic_dir / n) for n in
               ("slide_offsets.csv", "coordinates.npy", "mask_counts.csv.gz",
                "retrieval_population.csv", "COMPLETE.json")},
            "masks_sha256": {f"fold{f}_budget{b}": sha256_file(mosaic_dir / f"fold{f}_budget{b}_mask.npy")
                             for b in budgets for f in sorted(int(x) for x in folds)},
            "feature_pickles": "each verified against slide_offsets.feature_sha256 and coordinates.npy",
        },
        "code_sha256": {str(p): sha256_file(Path(p)) for p in code_paths},
        "bytes_read": read_bytes, "wall_seconds": None,
        "budgets": {}, "passthrough": pt_rows,
    }
    if qf:
        record["head_fold"] = head_fold
        record["design"] = ("query-fold head: head_fold filters every population slide; the bank is "
                            "fold head_fold's query source and reference library (ADR0021 section B)")
    suffix = "" if full_range else f"_{lo}_{hi or len(off)}"
    for b in budgets:
        r = rows[b]
        written = [x for x in r if x[9] == 1]
        skipped = [x for x in r if x[9] == 0]
        summ_sha = _write_csv(out_dirs[b] / f"_filter_summary{suffix}.csv",
                              SUMMARY_HEADER_QF if qf else SUMMARY_HEADER, r)
        record["budgets"][str(b)] = {
            "slides": len(r), "written": len(written), "skipped": len(skipped),
            "skipped_slides": [dict(slide_id=x[1], diagnostic_group=x[3], slide_fold=x[4],
                                    original_tiles=x[7], retained_tiles=x[8]) for x in skipped],
            "skipped_by_group": pd.Series([x[3] for x in skipped], dtype=object).value_counts().to_dict(),
            "original_tiles": int(sum(x[7] for x in r)),
            "retained_tiles_all": int(sum(x[8] for x in r)),
            "retained_tiles_written": int(sum(x[8] for x in written)),
            "filter_summary_sha256": summ_sha,
        }
        if qf:
            own = [x for x in r if x[13] == 1]
            record["budgets"][str(b)].update({
                "head_fold": head_fold, "library_size": len(written),
                "own_fold_queries_intended": len(own),
                "own_fold_queries_returned": sum(1 for x in own if x[9] == 1),
                "own_fold_non_returned": [x[1] for x in own if x[9] == 0],
                "library_excluded_slides": [x[1] for x in skipped],
            })
    record["wall_seconds"] = round(time.time() - t0, 1)
    for b in budgets:
        write_record(out_dirs[b] / f"_filter_record{suffix}.json",
                   json.dumps(record, indent=1, default=str).encode())
    if pt:
        write_record(out_root / "passthrough" / f"_passthrough_record{suffix}.json",
                   json.dumps(record, indent=1, default=str).encode())
    return record


UNFILTERED_LIBRARY_VIEW = "library_unfiltered_pop2604"


def _link_once(link: Path, target: Path) -> str:
    """Create ``link -> target`` unless an identical link exists; never replace anything."""
    if link.is_symlink() or link.exists():
        if not link.is_symlink() or os.readlink(link) != str(target):
            raise FilterError(f"{link} exists and is not a symlink to {target}; refusing to replace")
        return "existing_identical"
    os.symlink(str(target), str(link))
    return "linked"


def make_views(*, mosaic_dir: Path, features_dir: Path, banks_root: Path, views_root: Path,
               budgets: Sequence[int], head_folds: Sequence[int], unfiltered_library: bool = True,
               verify_sha: bool = True, code_paths: Sequence[Path] = ()) -> Dict:
    """Symlink directories that similarity_cli external mode reads (query-fold design).

    * ``queries/budget{b}/fold{F}/``: fold-F slides (own fold == F) returned under bank (b, F),
      linking to that bank's pickles -- the filtered query set of both filtered arms.
    * ``library_unfiltered_pop2604/``: every population slide's saved unfiltered pickle -- the
      reference library of the query-only arm (the saved bank restricted to the population).

    Each target is checked against the bank summary sha256 / accepted feature sha256
    (``verify_sha``) and every view's content equals its declared id set exactly."""
    t0 = time.time()
    off = load_offsets(mosaic_dir)
    pop = pd.read_csv(mosaic_dir / "retrieval_population.csv", dtype={"slide_id": str})
    if set(pop.slide_id) != set(off.slide_id):
        raise FilterError("retrieval_population and slide_offsets slide sets differ")
    rec = {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"), "argv": sys.argv,
           "verify_sha": verify_sha, "code_sha256": {str(c): sha256_file(Path(c)) for c in code_paths},
           "views": {}}
    for b in budgets:
        for f in head_folds:
            bank = banks_root / f"budget{b}" / f"fold{f}"
            summ = bank / "_filter_summary.csv"
            s = pd.read_csv(summ, dtype={"slide_id": str, "sha256": str})
            if set(s.slide_id) != set(off.slide_id) or not (s.head_fold == f).all():
                raise FilterError(f"{summ}: not a complete head-fold-{f} bank summary")
            own = s.slide_id.map(off.set_index("slide_id").slide_fold) == f
            if not (own.astype(int) == s.own_fold_query).all():
                raise FilterError(f"{summ}: own_fold_query disagrees with slide_offsets")
            on_disk = sorted(x.stem for x in bank.glob("*.pkl"))
            if on_disk != sorted(s.slide_id[s.written == 1]):
                raise FilterError(f"{bank}: pickles on disk differ from the summary's written set")
            q = s[(s.written == 1) & own].sort_values("slide_id")
            vdir = views_root / "queries" / f"budget{b}" / f"fold{f}"
            vdir.mkdir(parents=True, exist_ok=True)
            for sid, sha in zip(q.slide_id, q.sha256):
                tgt = bank / f"{sid}.pkl"
                if verify_sha and sha256_file(tgt) != sha:
                    raise FilterError(f"{tgt}: sha256 differs from {summ}")
                _link_once(vdir / f"{sid}.pkl", tgt)
            have = sorted(x.stem for x in vdir.glob("*.pkl"))
            if have != sorted(q.slide_id):
                raise FilterError(f"{vdir}: view content differs from the declared query set")
            rec["views"][f"queries/budget{b}/fold{f}"] = {
                "bank": str(bank), "bank_summary_sha256": sha256_file(summ),
                "queries_intended": int(own.sum()), "queries_returned": len(q),
                "non_returned_queries": sorted(s.slide_id[own & (s.written == 0)]),
                "library_size": int((s.written == 1).sum()),
                "library_excluded": sorted(s.slide_id[s.written == 0])}
    if unfiltered_library:
        vdir = views_root / UNFILTERED_LIBRARY_VIEW
        vdir.mkdir(parents=True, exist_ok=True)
        for row in off.itertuples(index=False):
            tgt = features_dir / f"{row.slide_id}.pkl"
            if verify_sha and sha256_file(tgt) != row.feature_sha256:
                raise FilterError(f"{tgt}: sha256 differs from slide_offsets.feature_sha256")
            _link_once(vdir / f"{row.slide_id}.pkl", tgt)
        have = sorted(x.stem for x in vdir.glob("*.pkl"))
        if have != sorted(off.slide_id):
            raise FilterError(f"{vdir}: view content differs from the population")
        rec["views"][UNFILTERED_LIBRARY_VIEW] = {"source": str(features_dir), "slides": len(have)}
    rec["wall_seconds"] = round(time.time() - t0, 1)
    views_root.mkdir(parents=True, exist_ok=True)
    rec["record"] = str(write_record(views_root / "_views_record.json",
                                     json.dumps(rec, indent=1, default=str).encode()))
    return rec


def main(argv=None) -> int:
    out = _default_out()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", type=int, nargs="*", default=[5, 100], choices=[5, 100])
    p.add_argument("--mosaic-dir", type=Path,
                   default=out / "r2_1_mosaic_cpu_20261003" / "attempt_001" if out else None)
    p.add_argument("--features-dir", type=Path,
                   default=out / "features" / "h_optimus_1" if out else None)
    p.add_argument("--out-root", type=Path, default=None,
                   help="default r2_1_filter_20261003/features (features_qf with --head-fold)")
    p.add_argument("--min-tiles", type=int, default=8)
    p.add_argument("--lo", type=int, default=1, help="1-based first slide (slide_offsets order)")
    p.add_argument("--hi", type=int, default=0, help="1-based last slide (0 = all)")
    p.add_argument("--passthrough", nargs="*", default=[],
                   help="slide ids to copy with an all-True mask into <out-root>/passthrough")
    p.add_argument("--head-fold", type=int, default=None,
                   help="query-fold design: filter every slide with this head's mask into "
                        "<out-root>/budget{b}/fold{F}/ (ADR0021 section B)")
    p.add_argument("--make-views", action="store_true",
                   help="query-fold design: build the similarity_cli symlink views (query sets per "
                        "budget/head fold and the unfiltered population library) from existing "
                        "--head-fold banks; writes nothing else")
    p.add_argument("--views-root", type=Path, default=None,
                   help="default r2_1_filter_20261003/similarity_inputs_qf")
    p.add_argument("--head-folds", type=int, nargs="*", default=[0, 1, 2, 3, 4],
                   help="--make-views: head folds")
    p.add_argument("--no-unfiltered-library", action="store_true",
                   help="--make-views: skip the unfiltered population library view")
    a = p.parse_args(argv)
    if a.make_views:
        banks = a.out_root or (out / "r2_1_filter_20261003" / "features_qf" if out else None)
        views = a.views_root or (out / "r2_1_filter_20261003" / "similarity_inputs_qf" if out else None)
        if banks is None or views is None or a.mosaic_dir is None or a.features_dir is None:
            p.error("export PROSPER_OUT or pass --mosaic-dir/--features-dir/--out-root/--views-root")
        rec = make_views(mosaic_dir=a.mosaic_dir, features_dir=a.features_dir, banks_root=banks,
                         views_root=views, budgets=a.budgets, head_folds=a.head_folds,
                         unfiltered_library=not a.no_unfiltered_library,
                         code_paths=[Path(__file__).resolve()])
        print(json.dumps(rec, indent=1, default=str))
        return 0
    if a.out_root is None and out:
        a.out_root = out / "r2_1_filter_20261003" / ("features" if a.head_fold is None else "features_qf")
    if a.mosaic_dir is None or a.features_dir is None or a.out_root is None:
        p.error("export PROSPER_OUT or pass --mosaic-dir/--features-dir/--out-root")
    rec = run(mosaic_dir=a.mosaic_dir, features_dir=a.features_dir, out_root=a.out_root,
              budgets=a.budgets, min_tiles=a.min_tiles, lo=a.lo, hi=a.hi,
              passthrough=a.passthrough, code_paths=[Path(__file__).resolve()],
              head_fold=a.head_fold)
    print(json.dumps({k: v for k, v in rec.items() if k != "passthrough"}, indent=1, default=str))
    print(json.dumps(rec["passthrough"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
