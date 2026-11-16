"""Aggregate tile features to one vector per slide, then build the distance matrix.

Ported from two `comet-analysis/src` scripts that cannot run on this cluster
(both hardcode the `/media/core` workstation mount):

  * ``median_pooling_multiprocess.py`` -- float16 cast, then per-dimension
    median over a slide's tiles.
  * ``create_sim_matrix_by_slide_features.py`` -- ``pdist(metric='cityblock')``,
    ``squareform``, ``np.fill_diagonal(1e6)``, saved as a scan-id-indexed
    DataFrame.

Three details are load-bearing and easy to get wrong:

1. **The published embedding matrices use L1 (`cityblock`), not cosine.**
   Only the OT branch is cosine-ground-cost. Using cosine here would produce a
   plausible matrix that silently is not comparable to any Table S3 row.
2. **The float16 cast happens before the median**, so it is part of the
   published definition rather than a storage detail.
3. **Same-patient masking is a separate stage** that the published pipeline did
   in a notebook (``update_sim_matrix_patient_id.ipynb``): same-``UID.Subject``
   off-diagonal cells are set to the 1e6 sentinel and the result is saved under
   a ``_no_same_patient`` filename. The evaluator's own flag is dead code, so a
   matrix that skips this stage is NOT patient-excluded no matter what the
   evaluator is told ([decision record]).

``--fidelity-encoder`` re-derives an already-published matrix from its cached
tile features and reports max/mean absolute deviation. Nothing new should be
believed before that passes.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from ..evaluation import paths

SENTINEL = 1e6
# Fixed so the nested tile ladder is reproducible across rungs and re-runs.
_LADDER_SEED = 20260827
# virchow2 features carry 1280 content dims plus extras; the published pooling
# script truncates. Kept as a table so a new encoder cannot silently inherit it.
TRUNCATE: Dict[str, int] = {"virchow2": 1280}

# SAMPLER, per comet-uni2/src/sampler_gen.py: a slide is represented by the
# per-dimension 5/15/.../95th percentiles of its tile features -- a (10, D)
# array, flattened here so it shares the same pdist/mask/save path as median
# pooling. Corrected 2026-08-28: this WAS assumed to have no float16 cast,
# reasoning from sampler_gen.py's exploratory snippet alone. Fidelity-checked
# against the published uni2/sampler matrix and it FAILED (max rel diff
# 3.4e-5, just over the 1e-6 bar) -- small enough to look like a rounding
# choice, not a wrong algorithm. `prosper.embeddings.slide_embedding.
# percentile_embedding.PercentileSlideEmbeddingGenerator` (already in this
# repo, unrelated to this file) casts to float16 BEFORE the percentile call,
# the same pattern median pooling uses -- that was the missing step.
_SAMPLER_PERCENTILES = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]


def _pool(feats: np.ndarray, method: str) -> np.ndarray:
    if method == "sampler":
        # float16 BEFORE the percentile call: see the module-level note above.
        return np.percentile(feats.astype(np.float16), _SAMPLER_PERCENTILES,
                             axis=0).ravel()
    # float16 BEFORE the median: part of the published definition.
    return np.median(feats.astype(np.float16), axis=0)


def _slide_vectors(feature_dir: Path, truncate: Optional[int],
                   fraction: Optional[float] = None,
                   method: str = "median_pooling"
                   ) -> Tuple[List[str], np.ndarray]:
    files = sorted(glob.glob(str(feature_dir / "*.pkl")))
    if not files:
        raise SystemExit(f"no feature pkls in {feature_dir}")
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    for fp in files:
        sid = os.path.basename(fp).split(".")[0]
        with open(fp, "rb") as fh:
            obj = pickle.load(fh)
        feats = obj[0] if isinstance(obj, tuple) and len(obj) == 2 else obj
        if feats is None or getattr(feats, "size", 0) == 0:
            print(f"[aggregate] {sid}: empty features, skipped", flush=True)
            continue
        feats = np.asarray(feats)
        if truncate:
            feats = feats[:, :truncate]
        if fraction is not None and fraction < 1.0:
            feats = _subsample(feats, sid, fraction)
        vecs.append(_pool(feats, method))
        ids.append(sid)
    return ids, np.vstack(vecs)


def _subsample(feats: np.ndarray, sid: str, fraction: float) -> np.ndarray:
    """A **nested** random subset of one slide's tiles.

    The rungs of the R2.2 ladder must be nested -- 15% subset of 30% subset of
    50% subset of 100% -- or the comparison confounds tile *count* with tile
    *identity*, which is precisely the confound that makes the published
    mosaic-vs-all-tiles comparison uninterpretable (standing rule 4: the two
    sets are not nested, pooled overlap 93.7%). Nesting is obtained by drawing
    ONE permutation per slide from a slide-specific seed and taking prefixes,
    so every fraction is a prefix of every larger one and the whole ladder is
    reproducible from the slide id alone.

    This isolates tile count *inside one tile universe*. It is deliberately NOT
    a reconstruction of the published 15% mosaic, which was a spatial-KMeans
    selection over a different (partial-tissue-admitting) universe with an
    unrecoverable seed.
    """
    n = feats.shape[0]
    n_keep = max(1, int(round(n * fraction)))
    if n_keep >= n:
        return feats
    # Seed from the slide id so the permutation is stable across rungs, across
    # jobs and across re-runs, without needing to persist an index file.
    seed = (int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16)
            ^ _LADDER_SEED)
    order = np.random.default_rng(seed).permutation(n)
    return feats[np.sort(order[:n_keep])]


def _distance_frame(ids: List[str], mat: np.ndarray,
                    metric: str) -> pd.DataFrame:
    d = squareform(pdist(mat, metric=metric))
    np.fill_diagonal(d, SENTINEL)
    return pd.DataFrame(d, index=ids, columns=ids)


def _mask_same_patient(df: pd.DataFrame, patient_col: str = "UID.Subject"
                       ) -> Tuple[pd.DataFrame, int]:
    """Demote same-patient off-diagonal cells to the 1e6 sentinel."""
    meta = pd.read_csv(paths.slide_metadata())
    id_col = "Slide.Scan.File"
    meta[id_col] = meta[id_col].astype(str)
    lookup = (meta.drop_duplicates(id_col)
                  .set_index(id_col)[patient_col].to_dict())
    ids = [str(i) for i in df.index]
    pat = np.array([str(lookup.get(i, f"__unknown_{i}")) for i in ids])
    same = pat[:, None] == pat[None, :]
    np.fill_diagonal(same, False)
    out = df.copy()
    vals = out.to_numpy()
    n_masked = int(same.sum())
    vals[same] = SENTINEL
    out.iloc[:, :] = vals
    return out, n_masked // 2


def build(feature_dir: Path, model: str, method: str, metric: str,
          out_dir: Path, truncate: Optional[int],
          fraction: Optional[float] = None) -> Dict:
    t0 = time.time()
    ids, mat = _slide_vectors(feature_dir, truncate, fraction, method)
    print(f"[aggregate] {len(ids)} slides -> {mat.shape} slide vectors "
          f"({time.time()-t0:.0f}s)", flush=True)
    raw = _distance_frame(ids, mat.astype(np.float64), metric)
    masked, n_pairs = _mask_same_patient(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"combined_similarity_matrix_{model}_{method}"
    with open(out_dir / f"{base}.pkl", "wb") as fh:
        pickle.dump(raw, fh)
    with open(out_dir / f"{base}_no_same_patient.pkl", "wb") as fh:
        pickle.dump(masked, fh)
    finite = raw.to_numpy()[raw.to_numpy() < SENTINEL]
    return {
        "model": model, "method": method, "metric": metric,
        "n_slides": len(ids), "feature_dim": int(mat.shape[1]),
        "truncate": truncate,
        "tile_fraction": fraction,
        "tile_fraction_nesting": ("prefixes of one per-slide permutation seeded "
                                  f"from sha256(slide_id) ^ {_LADDER_SEED}"
                                  if fraction is not None and fraction < 1.0
                                  else None),
        "same_patient_pairs_masked": n_pairs,
        "max_real_distance": float(finite.max()) if finite.size else None,
        "feature_dir": str(feature_dir),
        "out_dir": str(out_dir),
        "wall_seconds": round(time.time() - t0, 1),
    }


def build_external(query_dir: Path, library_dir: Path, model: str, method: str,
                   metric: str, out_dir: Path, truncate: Optional[int],
                   fraction: Optional[float] = None) -> Dict:
    """Query x library distance matrix for an EXTERNAL cohort (C3 / C12).

    Rectangular, unmasked (an external slide has no COMET patient), no
    ``_no_same_patient`` twin: the external evaluator takes the matrix path
    explicitly rather than resolving it through ``paths.similarity_matrix``.
    The library vectors are computed from the published tile features with
    the same pooling, cast and truncation as the square matrices, so a
    query's distances are on the same scale as Table S3's.
    """
    from scipy.spatial.distance import cdist
    t0 = time.time()
    q_ids, q_mat = _slide_vectors(query_dir, truncate, fraction, method)
    l_ids, l_mat = _slide_vectors(library_dir, truncate, None, method)
    d = cdist(q_mat.astype(np.float64), l_mat.astype(np.float64), metric=metric)
    df = pd.DataFrame(d, index=q_ids, columns=l_ids)
    overlap = [q for q in q_ids if q in set(l_ids)]
    for q in overlap:
        df.loc[q, q] = SENTINEL
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"combined_similarity_matrix_{model}_{method}"
    with open(out_dir / f"{base}.pkl", "wb") as fh:
        pickle.dump(df, fh)
    finite = d[d < SENTINEL]
    return {
        "model": model, "method": method, "metric": metric, "external": True,
        "n_query_slides": len(q_ids), "n_library_slides": len(l_ids),
        "n_query_ids_also_in_library": len(overlap),
        "feature_dim": int(q_mat.shape[1]), "truncate": truncate,
        "tile_fraction": fraction,
        "max_real_distance": float(finite.max()) if finite.size else None,
        "query_dir": str(query_dir), "library_dir": str(library_dir),
        "out_dir": str(out_dir), "wall_seconds": round(time.time() - t0, 1),
    }


def fidelity(encoder: str, method: str, metric: str,
             truncate: Optional[int]) -> int:
    """Re-derive a published matrix from its cached tile features."""
    comet = paths.comet_root() / f"comet-{encoder}" / "data" / "tmp_final"
    paths.require(comet, f"{encoder} tile features")
    published = (paths.comet_root() / f"comet-{encoder}" / "output"
                 / "similarity_matrix"
                 / f"combined_similarity_matrix_{encoder}_{method}.pkl")
    paths.require(published, "published matrix")
    ids, mat = _slide_vectors(comet, truncate, method=method)
    mine = _distance_frame(ids, mat.astype(np.float64), metric)
    with open(published, "rb") as fh:
        ref = pickle.load(fh)
    ref.index = [str(i) for i in ref.index]
    ref.columns = [str(c) for c in ref.columns]
    common = [i for i in mine.index if i in set(ref.index)]
    a = mine.loc[common, common].to_numpy()
    b = ref.loc[common, common].to_numpy()
    live = (a < SENTINEL) & (b < SENTINEL)
    diff = np.abs(a[live] - b[live])
    rel = diff / np.maximum(np.abs(b[live]), 1e-12)
    print(json.dumps({
        "encoder": encoder, "method": method, "metric": metric,
        "n_common_slides": len(common),
        "n_compared_cells": int(live.sum()),
        "max_abs_diff": float(diff.max()) if diff.size else None,
        "mean_abs_diff": float(diff.mean()) if diff.size else None,
        "max_rel_diff": float(rel.max()) if rel.size else None,
    }, indent=2))
    # float16 pooling then float64 pdist is deterministic, so an exact port
    # should land at machine precision. 1e-6 relative is a generous bar that
    # still fails loudly on a wrong metric (cityblock vs cosine differs by
    # orders of magnitude, not by 1e-6).
    ok = bool(rel.size and rel.max() < 1e-6)
    print("FIDELITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True)
    p.add_argument("--method", default="median_pooling")
    p.add_argument("--metric", default="cityblock",
                   help="published embedding matrices are L1 (cityblock)")
    p.add_argument("--features", default=None,
                   help="feature dir; default $PROSPER_OUT/features/<encoder>")
    p.add_argument("--matrix-model", default=None,
                   help="model token in the output filename (default --encoder)")
    p.add_argument("--tile-fraction", type=float, default=None,
                   help="keep this nested fraction of each slide's tiles "
                        "(R2.2 ladder). Omit or 1.0 for all tiles.")
    p.add_argument("--min-slides", type=int, default=0,
                   help="fail unless at least this many feature pkls are "
                        "present. Guards against building a matrix on a "
                        "half-finished extraction: this stage globs whatever "
                        "*.pkl it finds, so a partial feature dir yields a "
                        "smaller matrix that looks perfectly valid.")
    p.add_argument("--fidelity-encoder", default=None,
                   help="instead of building, re-derive this published "
                        "encoder's matrix and report deviation")
    p.add_argument("--library-features", default=None,
                   help="EXTERNAL mode: build a query x library matrix against "
                        "this reference-library feature dir instead of a "
                        "square self-matrix")
    a = p.parse_args(argv)

    truncate = TRUNCATE.get(a.fidelity_encoder or a.encoder)
    if a.fidelity_encoder:
        return fidelity(a.fidelity_encoder, a.method, a.metric, truncate)

    out_root = paths.output_root()
    feature_dir = Path(a.features) if a.features else (
        out_root / "features" / a.encoder)
    paths.require(feature_dir, f"{a.encoder} features")
    if a.min_slides:
        # An LSF `-w done(<array>)` chain cannot be relied on for this: one
        # failed array element makes `done` unsatisfiable and LSF silently kills
        # the whole downstream chain (this happened -- all 8 R2.2 ladder jobs
        # died with no exit code and no log when extraction element 32 hit the
        # memory limit). So the chain now uses `ended`, which fires regardless,
        # and completeness is checked HERE where it can fail loudly instead of
        # being assumed by the scheduler.
        n_have = len(glob.glob(str(feature_dir / "*.pkl")))
        if n_have < a.min_slides:
            raise SystemExit(
                f"[aggregate] REFUSING to build: {n_have} feature pkls in "
                f"{feature_dir}, expected >= {a.min_slides}. Extraction is "
                f"incomplete, and a matrix built now would be silently wrong "
                f"(fewer rows, different library) rather than obviously wrong.")
        print(f"[aggregate] completeness gate: {n_have} >= {a.min_slides} OK",
              flush=True)
    if a.library_features:
        prov = build_external(feature_dir,
                              paths.require(Path(a.library_features), "library features"),
                              a.matrix_model or a.encoder, a.method, a.metric,
                              out_root / "similarity_matrix", truncate,
                              a.tile_fraction)
    else:
        prov = build(feature_dir, a.matrix_model or a.encoder, a.method, a.metric,
                     out_root / "similarity_matrix", truncate, a.tile_fraction)
    log = out_root / "logs" / "aggregate"
    log.mkdir(parents=True, exist_ok=True)
    tag = f"{a.matrix_model or a.encoder}_{a.method}"
    (log / f"provenance_{tag}.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
