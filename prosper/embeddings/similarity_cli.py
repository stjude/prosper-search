"""CLI: pairwise slide-similarity rows for one range of query slides.

Covers the two ``PRODUCES_SIMILARITY`` methods in
``prosper.embeddings.slide_embedding`` -- OT (exact Wasserstein, cosine
ground cost) and BoB/Yottixel (median-of-min Hamming distance) -- through one
driver, since both generators already share an identical interface:
``compute_pairwise_similarities(slide_data, output_dir, slide_id=<one query>)``,
with the same per-query pickle cache and the same ``1e6`` self-match sentinel.
Median pooling and SAMPLER are NOT here; they collapse to one vector per
slide and go through ``aggregate_cli.py``'s ``pdist`` path instead, which is
orders of magnitude cheaper.

One LSF array element handles a contiguous range of QUERY slides (indices
into the feature directory's own sorted file list -- the same convention
``aggregate_cli.py`` uses, so the reference-library order matches what the
matrix gets built with downstream and [decision record]'s fix isn't silently
reintroduced by a different sort). Every query in an element is compared
against the FULL reference library, which must be loaded once regardless of
how many queries the element handles -- so wider ranges amortize that load
over more queries. Canary before choosing a range width (rule 7): the
per-query cost is method- and library-size-dependent, and for OT at mosaic
scale the published reference (``comet-uni2/src/compute_ot_similarities_lsf.py``)
needed up to a 24h wall-clock limit for a SINGLE query slide on 1 core.

``--num-workers`` parallelises the reference-library loop within one query
for OT specifically (the LP solve per pair is embarrassingly parallel and
releases the GIL in the C extension); BoB's inner loop is already one BLAS
matmul per query and does not benefit from this.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..evaluation import paths
from .aggregate_cli import _subsample
from .slide_embedding import get_slide_embedding_generator
from .slide_embedding.bob_embedding import hamming_min_median

_SELF_SENTINEL = 1e6


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load_slide_data(feature_dir: Path, generator, truncate: Optional[int],
                     fraction: Optional[float] = None, keep: Optional[set] = None
                     ) -> Tuple[List[str], Dict[str, Dict]]:
    """Load per-slide features. With ``keep`` (external query side, C3 2026-08-30)
    only those slide ids are read into memory; every id is still returned so the
    ``--lo/--hi`` index space is the sorted glob order the drivers size their
    arrays from. Loading all 4,160 CCDI query slides beside the 2,921-slide
    library was a TERM_MEMLIMIT kill at 41 GB for a 3-row canary."""
    import glob
    files = sorted(glob.glob(str(feature_dir / "*.pkl")))
    if not files:
        raise SystemExit(f"no feature pkls in {feature_dir}")
    ids: List[str] = []
    slide_data: Dict[str, Dict] = {}
    for fp in files:
        sid = os.path.basename(fp).split(".")[0]
        if keep is not None and sid not in keep:
            ids.append(sid)
            continue
        with open(fp, "rb") as fh:
            obj = pickle.load(fh)
        feats = obj[0] if isinstance(obj, tuple) and len(obj) == 2 else obj
        if feats is None or getattr(feats, "size", 0) == 0:
            print(f"[similarity] {sid}: empty features, skipped", flush=True)
            continue
        feats = np.asarray(feats)
        if truncate:
            feats = feats[:, :truncate]
        if fraction is not None and fraction < 1.0:
            # R2.2 ladder for the set-based aggregations (C7c): the SAME nested
            # per-slide prefix aggregate_cli uses, so a BoB/OT rung at 15% is
            # built on exactly the tiles the median-pooling rung at 15% used.
            feats = _subsample(feats, sid, fraction)
        processed = generator.process_single_slide(feats, sid)
        # OT's process_single_slide keeps BOTH the normalised and the raw
        # array ('features_raw') even though only 'features' is ever read
        # downstream here -- doubling the whole-library memory footprint for
        # no benefit. Measured: this is what turned a real-library canary
        # into a TERM_MEMLIMIT kill at 16 GB/element (2,921 slides x ~1,300
        # tiles x 1536-d x 4 bytes x 2 copies). Drop it; BoB's dict has no
        # such duplicate to begin with.
        processed.pop("features_raw", None)
        slide_data[sid] = processed
        ids.append(sid)
    return ids, slide_data


def _compute_query_row_parallel(generator, slide_data: Dict[str, Dict],
                                qid: str, other_ids: List[str],
                                num_workers: int) -> List[Tuple[str, float]]:
    """OT's inner loop, parallelised. Same math as
    ``OTSlideEmbeddingGenerator.compute_pairwise_similarities``'s serial
    loop, just distributed over ``num_workers`` threads.

    Deliberately ``threading``, not a process pool: the reference library
    (``slide_data``) is the whole feature set for this element, tens of GB at
    mosaic scale, and a process backend would re-pickle and ship a copy of it
    per call. Threads share memory, so there is no serialisation cost either
    way -- worst case (POT's EMD solve does not release the GIL) this is no
    faster than serial, but it is never slower or wrong, which a
    naively-applied process pool could be here.
    """
    from joblib import Parallel, delayed

    q = slide_data[qid]["features"]

    def _one(oid: str) -> float:
        if oid == qid:
            return _SELF_SENTINEL
        try:
            return float(generator.compute_ot_distance(q, slide_data[oid]["features"]))
        except Exception as exc:                     # noqa: BLE001
            print(f"[similarity] {qid} vs {oid}: FAILED {exc}", flush=True)
            return _SELF_SENTINEL

    dists = Parallel(n_jobs=num_workers, backend="threading")(
        delayed(_one)(oid) for oid in other_ids)
    return list(zip(other_ids, dists))


def _row_vs_library(generator, method: str, qid: str, qdata: Dict,
                    library: Dict[str, Dict], library_ids: List[str],
                    num_workers: int) -> List[Tuple[str, float]]:
    """One EXTERNAL query against a fixed reference library (C3 / C12).

    The self-library path above computes every query against the same
    directory it was loaded from; an external cohort's queries live in a
    different directory from the COMET library, and a query-vs-query block
    would be wasted work (and would not be a COMET retrieval). Same math as
    the generators' own loops; only the candidate set differs. A library slide
    with the query's own id (cannot happen for a genuinely external cohort;
    happens in the self-check fidelity mode) gets the self sentinel.
    """
    if method == "ot":
        merged = dict(library)
        merged[qid] = qdata
        return _compute_query_row_parallel(generator, merged, qid, library_ids,
                                           max(1, num_workers))
    out: List[Tuple[str, float]] = []
    q = qdata["barcodes"]
    for oid in library_ids:
        if oid == qid:
            out.append((oid, _SELF_SENTINEL))
            continue
        try:
            out.append((oid, float(hamming_min_median(q, library[oid]["barcodes"]))))
        except Exception as exc:                     # noqa: BLE001
            print(f"[similarity] {qid} vs {oid}: FAILED {exc}", flush=True)
            out.append((oid, _SELF_SENTINEL))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True)
    p.add_argument("--method", required=True, choices=["ot", "bob"])
    p.add_argument("--features", default=None,
                   help="feature dir; default $PROSPER_OUT/features/<encoder>")
    p.add_argument("--feature-dim", type=int, required=True,
                   help="native feature width for this encoder. OT truncates "
                        "to this; BoB does not apply it to the computation "
                        "but still records it. Passing the wrong value "
                        "silently truncates a new encoder's features -- "
                        "there is no safe default.")
    p.add_argument("--metric", default="cosine", help="OT ground cost only")
    p.add_argument("--reg", type=float, default=0.0,
                   help="OT entropic regularisation. 0 = exact EMD, the "
                        "published definition. Nonzero is Sinkhorn and does "
                        "not reproduce Table S3 (see open item in NEXT.md).")
    p.add_argument("--lo", type=int, default=1, help="1-based first query index")
    p.add_argument("--hi", type=int, default=0, help="1-based last (0 = all)")
    p.add_argument("--num-workers", type=int, default=1,
                   help="OT only: parallelise one query's reference-library "
                        "loop over this many processes")
    p.add_argument("--task", default=None,
                   help="output subdir; default similarity/<encoder>_<method>")
    p.add_argument("--library-features", default=None,
                   help="EXTERNAL mode: reference-library feature dir (e.g. the "
                        "published comet-uni2/data/tmp_final). Queries then come "
                        "from --features only and rows are query x library.")
    p.add_argument("--tile-fraction", type=float, default=None,
                   help="R2.2 ladder (C7c): nested per-slide tile prefix, "
                        "identical to aggregate_cli --tile-fraction")
    a = p.parse_args(argv)

    out_root = paths.output_root()
    feature_dir = Path(a.features) if a.features else (
        out_root / "features" / a.encoder)
    paths.require(feature_dir, f"{a.encoder} features")
    task = a.task or f"similarity/{a.encoder}_{a.method}"
    out_dir = out_root / task
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = dict(feature_dim=a.feature_dim, verbose=False)
    if a.method == "ot":
        gen_kwargs.update(metric=a.metric, reg=a.reg)
    generator = get_slide_embedding_generator(a.method, **gen_kwargs)
    trunc = a.feature_dim if a.method == "ot" else None

    t0 = time.time()
    external = a.library_features is not None
    if external:
        lib_dir = paths.require(Path(a.library_features), "library features")
        ids, slide_data = _load_slide_data(lib_dir, generator, trunc, a.tile_fraction)
        import glob
        all_q = [os.path.basename(f).split(".")[0]
                 for f in sorted(glob.glob(str(feature_dir / "*.pkl")))]
        want = set(all_q[max(1, a.lo) - 1:(a.hi or len(all_q))])
        q_ids, q_data = _load_slide_data(feature_dir, generator, trunc, a.tile_fraction, keep=want)
    else:
        ids, slide_data = _load_slide_data(feature_dir, generator, trunc, a.tile_fraction)
        q_ids, q_data = ids, slide_data
    load_s = time.time() - t0
    print(f"[similarity] loaded library {len(ids)} slides"
          f"{f' + {len(q_data)} of {len(q_ids)} external queries' if external else ''} "
          f"in {load_s:.0f}s", flush=True)

    hi = a.hi or len(q_ids)
    lo = max(1, a.lo)
    queries = [q for q in q_ids[lo - 1:hi] if q in q_data]   # empty-feature slides are skipped, not shifted
    print(f"[similarity] method={a.method} encoder={a.encoder} "
          f"queries {lo}..{hi} of {len(q_ids)} ({len(queries)} in this element)",
          flush=True)

    done = 0
    t1 = time.time()
    for qid in queries:
        out_path = out_dir / f"similarity_{qid}.pkl"
        if out_path.exists():
            done += 1
            continue
        ts = time.time()
        if external:
            row = _row_vs_library(generator, a.method, qid, q_data[qid],
                                  slide_data, ids, a.num_workers)
            with open(out_path, "wb") as fh:
                pickle.dump(row, fh)
        elif a.method == "ot" and a.num_workers > 1:
            others = [o for o in ids]
            row = _compute_query_row_parallel(generator, slide_data, qid,
                                              others, a.num_workers)
            with open(out_path, "wb") as fh:
                pickle.dump(row, fh)
        else:
            generator.compute_pairwise_similarities(slide_data, str(out_dir),
                                                     slide_id=qid)
        done += 1
        print(f"[similarity] {qid}: row done in {time.time()-ts:.1f}s", flush=True)

    wall = time.time() - t1
    prov = {
        "task": task, "encoder": a.encoder, "method": a.method,
        "feature_dim": a.feature_dim,
        "metric": a.metric if a.method == "ot" else None,
        "reg": a.reg if a.method == "ot" else None,
        "external": external,
        "query_features": str(feature_dir),
        "library_features": str(a.library_features) if external else str(feature_dir),
        "tile_fraction": a.tile_fraction,
        "tile_fraction_nesting": ("prefixes of one per-slide permutation, as "
                                  "aggregate_cli" if a.tile_fraction else None),
        "n_library": len(ids), "n_queries_total": len(q_ids),
        "query_range": [lo, hi],
        "n_queries_this_element": len(queries), "n_done": done,
        "load_seconds": round(load_s, 1), "wall_seconds": round(wall, 1),
        "num_workers": a.num_workers,
        "git_commit": _git_commit(),
        "lsf_job": os.environ.get("LSB_JOBID"),
        "lsf_index": os.environ.get("LSB_JOBINDEX"),
    }
    pdir = out_root / "logs" / task.replace("/", "_")
    pdir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("LSB_JOBINDEX") or f"{lo}_{hi}"
    (pdir / f"provenance_{tag}.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
