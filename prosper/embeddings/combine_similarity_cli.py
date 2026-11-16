"""CLI: assemble per-query similarity rows (from ``similarity_cli.py``) into
one combined distance matrix, same-patient-masked exactly like
``aggregate_cli.py``'s median-pooling path.

Published filenames use ``ot_cosine`` / ``yottixel`` as the method token,
not the generator keys ``ot`` / ``bob`` that ``similarity_cli.py`` and the
``slide_embedding`` registry use -- ``OUTPUT_METHOD_TAG`` bridges the two so
``paths.similarity_matrix()`` resolves what this writes.

``--fidelity-reference`` re-derives a PUBLISHED matrix's rows from a small
probe set and reports deviation, mirroring ``aggregate_cli.py --fidelity-encoder``.
Exact OT is too expensive to fidelity-check by rebuilding the whole matrix
(that IS the ~2,921-element, up-to-24h-per-row job this exists to avoid
running twice), so the probe is a handful of query rows compared against
their published counterparts -- catches an indexing, normalisation or
ground-cost mistake exactly as loudly as a full rebuild would, at a
trivial fraction of the cost.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..evaluation import paths
from .aggregate_cli import SENTINEL, _mask_same_patient

OUTPUT_METHOD_TAG: Dict[str, str] = {"ot": "ot_cosine", "bob": "yottixel"}


def _load_rows(similarity_dir: Path) -> Tuple[List[str], List[str], np.ndarray]:
    files = sorted(glob.glob(str(similarity_dir / "similarity_*.pkl")))
    if not files:
        raise SystemExit(f"no similarity_*.pkl in {similarity_dir}")
    row_by_qid: Dict[str, Dict[str, float]] = {}
    ref_ids: Optional[List[str]] = None
    for fp in files:
        qid = os.path.basename(fp)[len("similarity_"):-len(".pkl")]
        with open(fp, "rb") as fh:
            row = pickle.load(fh)  # list of (other_id, distance)
        row_ids = sorted(str(o) for o, _ in row)
        if ref_ids is None:
            ref_ids = row_ids
        elif row_ids != ref_ids:
            raise SystemExit(
                f"[combine] {qid}: reference-library mismatch vs the first "
                f"row seen -- {len(row_ids)} ids here, {len(ref_ids)} "
                f"expected. Every row must be computed against the same "
                f"library or the matrix is not square/comparable.")
        row_by_qid[qid] = {str(o): float(d) for o, d in row}
    qids = sorted(row_by_qid)
    mat = np.full((len(qids), len(ref_ids)), SENTINEL, dtype=np.float64)
    for i, qid in enumerate(qids):
        r = row_by_qid[qid]
        for j, oid in enumerate(ref_ids):
            mat[i, j] = r.get(oid, SENTINEL)
    return qids, ref_ids, mat


def build(similarity_dir: Path, model: str, tag: str, out_dir: Path,
          external: bool = False) -> Dict:
    t0 = time.time()
    qids, ref_ids, mat = _load_rows(similarity_dir)
    df = pd.DataFrame(mat, index=qids, columns=ref_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"combined_similarity_matrix_{model}_{tag}"
    with open(out_dir / f"{base}.pkl", "wb") as fh:
        pickle.dump(df, fh)
    if external:
        # Rectangular query x library block for an external cohort: no COMET
        # patient to mask and no `_no_same_patient` twin (the external
        # evaluator takes the matrix path explicitly).
        n_pairs = None
    else:
        if qids != ref_ids:
            raise SystemExit("[combine] rows are not square; pass --external "
                             "for a query x library block")
        masked, n_pairs = _mask_same_patient(df)
        with open(out_dir / f"{base}_no_same_patient.pkl", "wb") as fh:
            pickle.dump(masked, fh)
    finite = df.to_numpy()[df.to_numpy() < SENTINEL]
    return {
        "model": model, "method": tag, "external": external,
        "n_query_slides": len(qids), "n_reference_slides": len(ref_ids),
        "square": qids == ref_ids,
        "same_patient_pairs_masked": n_pairs,
        "max_real_distance": float(finite.max()) if finite.size else None,
        "similarity_dir": str(similarity_dir), "out_dir": str(out_dir),
        "wall_seconds": round(time.time() - t0, 1),
    }


def fidelity_reference(encoder: str, tag: str, similarity_dir: Path) -> int:
    """Compare this run's rows against the PUBLISHED combined matrix, for
    whatever query slides ``similarity_dir`` actually has -- a probe, not a
    full rebuild."""
    qids, ref_ids, mat = _load_rows(similarity_dir)
    mine = pd.DataFrame(mat, index=qids, columns=ref_ids)
    published = (paths.comet_root() / f"comet-{encoder}" / "output"
                / "similarity_matrix"
                / f"combined_similarity_matrix_{encoder}_{tag}.pkl")
    paths.require(published, "published matrix")
    with open(published, "rb") as fh:
        ref = pickle.load(fh)
    ref.index = [str(i) for i in ref.index]
    ref.columns = [str(c) for c in ref.columns]
    common_cols = [c for c in mine.columns if c in set(ref.columns)]
    common_rows = [r for r in mine.index if r in set(ref.index)]
    a = mine.loc[common_rows, common_cols].to_numpy()
    b = ref.loc[common_rows, common_cols].to_numpy()
    live = (a < SENTINEL) & (b < SENTINEL)
    diff = np.abs(a[live] - b[live])
    rel = diff / np.maximum(np.abs(b[live]), 1e-12)
    result = {
        "encoder": encoder, "method": tag,
        "n_probe_rows": len(common_rows), "n_common_cols": len(common_cols),
        "n_compared_cells": int(live.sum()),
        "max_abs_diff": float(diff.max()) if diff.size else None,
        "mean_abs_diff": float(diff.mean()) if diff.size else None,
        "max_rel_diff": float(rel.max()) if rel.size else None,
    }
    print(json.dumps(result, indent=2))
    # Both OT (float64 LP solve) and BoB (integer Hamming counts) are exact,
    # deterministic recomputations given the same inputs, so 1e-6 relative is
    # the same generous-but-not-blind bar aggregate_cli.py's fidelity uses.
    ok = bool(rel.size and rel.max() < 1e-6)
    print("FIDELITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True)
    p.add_argument("--method", required=True, choices=["ot", "bob"])
    p.add_argument("--similarity-dir", default=None,
                   help="default $PROSPER_OUT/similarity/<encoder>_<method>")
    p.add_argument("--matrix-model", default=None,
                   help="model token in the output filename (default --encoder)")
    p.add_argument("--min-slides", type=int, default=0,
                   help="fail unless at least this many query rows are present")
    p.add_argument("--fidelity-reference", action="store_true",
                   help="instead of building, compare this run's rows against "
                        "the published matrix and report deviation")
    p.add_argument("--external", action="store_true",
                   help="rows are external queries x the COMET library: build "
                        "the rectangular block, skip same-patient masking")
    a = p.parse_args(argv)

    out_root = paths.output_root()
    similarity_dir = Path(a.similarity_dir) if a.similarity_dir else (
        out_root / "similarity" / f"{a.encoder}_{a.method}")
    paths.require(similarity_dir, f"{a.encoder}/{a.method} similarity rows")
    tag = OUTPUT_METHOD_TAG[a.method]

    if a.fidelity_reference:
        return fidelity_reference(a.encoder, tag, similarity_dir)

    if a.min_slides:
        n_have = len(glob.glob(str(similarity_dir / "similarity_*.pkl")))
        if n_have < a.min_slides:
            raise SystemExit(
                f"[combine] REFUSING to build: {n_have} rows in "
                f"{similarity_dir}, expected >= {a.min_slides}.")
        print(f"[combine] completeness gate: {n_have} >= {a.min_slides} OK",
              flush=True)

    prov = build(similarity_dir, a.matrix_model or a.encoder, tag,
                 out_root / "similarity_matrix", external=a.external)
    log = out_root / "logs" / "combine_similarity"
    log.mkdir(parents=True, exist_ok=True)
    (log / f"provenance_{a.matrix_model or a.encoder}_{tag}.json").write_text(
        json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
