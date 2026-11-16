"""S2 (R2.6) -- tumour-only TITAN/CONCH: tile scoring and CONCH patch subsetting.

The published tumour-only experiment applied a tile classifier (SGD log-loss on
UNI2 / Virchow2 tile embeddings, classes 1..5 = Tumor, Immune cells, Necrosis,
nonTumor, Artifacts; ``comet-annotation``) to the *mosaic* tiles only. TITAN
consumes CONCH v1.5 features of 1024-px patches at 40x, so a tumour-only TITAN
slide embedding needs a tumour flag for **every** 1024-px patch. Two actions:

``score``   run the UNI2 tile classifier over the all-patches UNI2 features
            (``features/uni2_all_patches/<slide>.pkl`` = (N x 1536, coords)) and
            write per-slide ``<slide>.csv.gz`` with the predicted class and the
            tumour probability of each 512-px tile.

``subset``  for each slide, read the CONCH h5 (``coords`` (M x 2) level-0 x,y of
            1024-px patches; ``features`` (M x 768)), flag a patch as tumour when
            at least ``--min-tumour-tiles`` of its four 512-px tiles are predicted
            Tumor (plan v5 R2.6: >= 50% -> 2 of 4), and write the kept rows to a
            new trident job directory with the h5 attributes preserved, so
            ``run_batch_of_slides.py --task feat --slide_encoder titan`` can
            encode the subset unchanged. Slides with fewer than ``--min-patches``
            tumour patches are written to the exclusion list, not encoded.

``matrix``  read the TITAN slide embeddings (``slide_features_titan/<slide>.h5``,
            768-d) of a job directory and write the cosine distance matrix and
            its same-patient-masked twin under ``$PROSPER_OUT/similarity_matrix/``
            as ``combined_similarity_matrix_<model>_<method>[_tumor_only]{,_no_same_patient}.pkl``
            so the frozen evaluators resolve it by ``--model/--method [--tumor_only]``.

``filter``  ST2.0 phase 1 (spec 0002, [analysis plan]): emit class-filtered COPIES of
            ``(features, coords)`` feature pickles so the stock retrieval path
            (``similarity_cli --library-features``) can run a tumour-only or
            minus-Artifacts arm -- nothing in that path can mask tiles, so the
            mask has to happen at the feature files. Arms: ``passthrough``
            (identity round-trip through this writer; its retrieval rows are
            the fidelity gate against the unfiltered originals),
            ``tumor_only`` (pred == 1), ``minus_artifacts`` (pred != 5, written
            only when its keep-set differs from tumour-only's for that slide).
            Scores come from a prior ``score`` run over the same pickles; rows
            are verified aligned by their ``wsi_loc`` sequence before any mask
            is applied.

Tiles that the all-patches universe does not contain (partial-tissue tiles,
``tissue_coverage < 1``) count as non-tumour, which is conservative for the
flag; the summary records per slide how many of a patch's four tiles were scored.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation import paths

CLASS_NAMES = {1: "Tumor", 2: "Immune cells", 3: "Necrosis", 4: "nonTumor", 5: "Artifacts"}
TUMOR = 1


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_clf(p: Path):
    import joblib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = joblib.load(p)
    return clf


def score(a) -> dict:
    clf = _load_clf(a.classifier)
    classes = [int(c) for c in clf.classes_]
    if TUMOR not in classes:
        raise SystemExit(f"classifier classes {classes} lack the Tumor class {TUMOR}")
    ti = classes.index(TUMOR)
    files = sorted(Path(a.features_dir).glob("*.pkl"))
    files = files[a.lo - 1:a.hi] if a.hi else files[a.lo - 1:]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done, t0 = 0, time.time()
    for f in files:
        dst = out / f"{f.stem}.csv.gz"
        if dst.exists() and not a.overwrite:
            continue
        feats, coords = pd.read_pickle(f)
        X = np.asarray(feats, dtype=np.float32)
        if X.shape[1] != getattr(clf, "n_features_in_", X.shape[1]):
            raise SystemExit(f"{f.name}: feature dim {X.shape[1]} != classifier {clf.n_features_in_}")
        prob = clf.predict_proba(X)
        pred = np.asarray(classes)[prob.argmax(axis=1)]
        xy = np.asarray([c["wsi_loc"] for c in coords], dtype=np.int64)
        df = pd.DataFrame({"wsi_loc_x": xy[:, 0], "wsi_loc_y": xy[:, 1],
                           "pred": pred, "p_tumor": prob[:, ti].astype(np.float32)})
        df.to_csv(dst, index=False, compression="gzip")
        done += 1
    rec = {"action": "score", "classifier": str(a.classifier), "classifier_sha256_16": _sha(a.classifier),
           "classes": classes, "class_names": CLASS_NAMES, "features_dir": str(a.features_dir),
           "lo": a.lo, "hi": a.hi, "n_written": done, "elapsed_s": round(time.time() - t0, 1)}
    with open(out / f"_score_{a.lo}_{a.hi or 'end'}.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def subset(a) -> dict:
    import h5py
    scores = Path(a.scores_dir)
    conch = Path(a.conch_dir)
    job = Path(a.out_job_dir) / a.coords_dir
    dst_dir = job / f"features_{a.patch_encoder}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []
    off = np.array([[0, 0], [a.tile, 0], [0, a.tile], [a.tile, a.tile]], dtype=np.int64)
    for h5 in sorted(conch.glob("*.h5")):
        sid = h5.stem
        sf = scores / f"{sid}.csv.gz"
        if not sf.exists():
            excluded.append({"slide": sid, "reason": "no tile scores"})
            continue
        s = pd.read_csv(sf)
        tum = {(int(x), int(y)) for x, y, p in zip(s.wsi_loc_x, s.wsi_loc_y, s.pred) if p == TUMOR}
        scored = {(int(x), int(y)) for x, y in zip(s.wsi_loc_x, s.wsi_loc_y)}
        with h5py.File(h5, "r") as h:
            coords = h["coords"][:]
            feats = h["features"][:]
            c_attrs = dict(h["coords"].attrs)
            f_attrs = dict(h["features"].attrs)
        n_tum = np.zeros(len(coords), dtype=np.int8)
        n_scored = np.zeros(len(coords), dtype=np.int8)
        for i, (x, y) in enumerate(coords):
            for dx, dy in off:
                key = (int(x) + int(dx), int(y) + int(dy))
                n_scored[i] += key in scored
                n_tum[i] += key in tum
        keep = n_tum >= a.min_tumour_tiles
        rec = {"slide": sid, "n_patches": int(len(coords)), "n_kept": int(keep.sum()),
               "frac_kept": float(keep.mean()) if len(coords) else 0.0,
               "mean_tiles_scored_per_patch": float(n_scored.mean()) if len(coords) else 0.0}
        rows.append(rec)
        if keep.sum() < a.min_patches:
            excluded.append({"slide": sid, "reason": f"{int(keep.sum())} tumour patches < {a.min_patches}"})
            continue
        with h5py.File(dst_dir / f"{sid}.h5", "w") as h:
            d = h.create_dataset("coords", data=coords[keep])
            d.attrs.update(c_attrs)
            d = h.create_dataset("features", data=feats[keep])
            d.attrs.update(f_attrs)
    summary = pd.DataFrame(rows)
    summary.to_csv(job / "_tumour_patch_summary.csv", index=False)
    pd.DataFrame(excluded).to_csv(job / "_excluded_slides.csv", index=False)
    rec = {"action": "subset", "scores_dir": str(scores), "conch_dir": str(conch),
           "out": str(dst_dir), "min_tumour_tiles_of_4": a.min_tumour_tiles,
           "min_patches": a.min_patches, "n_slides_in": int(len(list(conch.glob('*.h5')))),
           "n_written": int(len(rows) - sum(1 for e in excluded if "tumour patches" in e["reason"])),
           "n_excluded": len(excluded), "median_frac_kept": float(summary.frac_kept.median()) if len(summary) else None}
    with open(job / "_subset_record.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def matrix(a) -> dict:
    import h5py
    import pickle
    from ..embeddings.aggregate_cli import _distance_frame, _mask_same_patient, SENTINEL
    src = Path(a.slide_features_dir)
    ids, vecs = [], []
    for h5 in sorted(src.glob("*.h5")):
        with h5py.File(h5, "r") as h:
            v = np.asarray(h["features"][:], dtype=np.float64).reshape(-1)
        ids.append(h5.stem)
        vecs.append(v)
    if len(ids) < a.min_slides:
        raise SystemExit(f"REFUSING: {len(ids)} slide embeddings < --min-slides {a.min_slides}")
    mat = np.stack(vecs)
    raw = _distance_frame(ids, mat, a.metric)
    masked, n_pairs = _mask_same_patient(raw)
    out_dir = paths.output_root() / "similarity_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"combined_similarity_matrix_{a.model}_{a.method}{'_tumor_only' if a.tumor_only else ''}"
    with open(out_dir / f"{base}.pkl", "wb") as fh:
        pickle.dump(raw, fh)
    with open(out_dir / f"{base}_no_same_patient.pkl", "wb") as fh:
        pickle.dump(masked, fh)
    finite = raw.to_numpy()[raw.to_numpy() < SENTINEL]
    rec = {"action": "matrix", "slide_features_dir": str(src), "n_slides": len(ids),
           "feature_dim": int(mat.shape[1]), "metric": a.metric, "model": a.model,
           "method": a.method, "tumor_only": a.tumor_only, "same_patient_pairs_masked": n_pairs,
           "max_real_distance": float(finite.max()) if finite.size else None,
           "out": str(out_dir / f"{base}_no_same_patient.pkl")}
    with open(out_dir / f"_{base}_record.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


ARMS = ("passthrough", "tumor_only", "minus_artifacts")


def _keep_mask(arm: str, pred: np.ndarray) -> np.ndarray:
    if arm == "passthrough":
        return np.ones(len(pred), dtype=bool)
    if arm == "tumor_only":
        return pred == TUMOR
    if arm == "minus_artifacts":
        return pred != 5
    raise SystemExit(f"unknown arm {arm}")


def filter_features(a) -> dict:
    import pickle
    src = Path(a.features_dir)
    scores = Path(a.scores_dir)
    out_root = Path(a.out_root)
    arms = [s.strip() for s in a.arms.split(",")]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {ARMS}")
        (out_root / arm).mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(src.glob("*.pkl")):
        sid = f.stem
        sf = scores / f"{sid}.csv.gz"
        if not sf.exists():
            raise SystemExit(f"{sid}: no score file {sf} -- run `score` first")
        with open(f, "rb") as fh:
            feats, coords = pickle.load(fh)
        feats = np.asarray(feats)
        s = pd.read_csv(sf)
        if len(s) != len(feats):
            raise SystemExit(f"{sid}: {len(s)} score rows != {len(feats)} tiles")
        xy = np.asarray([c["wsi_loc"] for c in coords], dtype=np.int64)
        if not (np.array_equal(xy[:, 0], s.wsi_loc_x.to_numpy())
                and np.array_equal(xy[:, 1], s.wsi_loc_y.to_numpy())):
            raise SystemExit(f"{sid}: score rows are not aligned with the pickle's coords")
        pred = s.pred.to_numpy()
        rec = {"slide": sid, "n_tiles": int(len(pred))}
        for c, name in CLASS_NAMES.items():
            rec[f"n_{name.split()[0].lower()}"] = int((pred == c).sum())
        masks = {arm: _keep_mask(arm, pred) for arm in arms}
        skip_equiv = ("minus_artifacts" in masks and "tumor_only" in masks
                      and np.array_equal(masks["minus_artifacts"], masks["tumor_only"]))
        for arm in arms:
            keep = masks[arm]
            rec[f"kept_{arm}"] = int(keep.sum())
            if arm == "minus_artifacts" and skip_equiv:
                rec["minus_artifacts_equiv_tumor_only"] = True
                continue
            if keep.sum() == 0:
                if getattr(a, "skip_empty", False):
                    # ST2.0 phase 2: a library slide with no predicted-Tumor tile (normal
                    # tissue, MISC) simply leaves the tumour-only library; record and go on.
                    print(f"[filter] SKIP {sid}/{arm}: 0 tiles kept (--skip-empty)", flush=True)
                    rec[f"skipped_{arm}"] = True
                    continue
                raise SystemExit(f"{sid}: arm {arm} keeps 0 tiles -- refusing to write")
            if keep.sum() < a.min_tiles:
                print(f"[filter] WARNING {sid}/{arm}: only {int(keep.sum())} tiles kept "
                      f"(< --min-tiles {a.min_tiles}); written anyway, flagged", flush=True)
                rec[f"below_min_{arm}"] = True
            dst = out_root / arm / f"{sid}.pkl"
            if dst.exists() and not a.overwrite:
                continue
            with open(dst, "wb") as fh:
                pickle.dump((feats[keep], [coords[i] for i in np.flatnonzero(keep)]), fh)
        rows.append(rec)
    summary = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / "_filter_summary.csv", index=False)
    rec = {"action": "filter", "features_dir": str(src), "scores_dir": str(scores),
           "out_root": str(out_root), "arms": arms, "min_tiles": a.min_tiles,
           "n_slides": len(rows),
           "n_minus_artifacts_written": int(sum(1 for r in rows
                                                if not r.get("minus_artifacts_equiv_tumor_only"))) if "minus_artifacts" in arms else None,
           "class_names": CLASS_NAMES}
    with open(out_root / "_filter_record.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("score")
    s.add_argument("--features-dir", type=Path, default=paths.output_root() / "features" / "uni2_all_patches")
    s.add_argument("--classifier", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, default=paths.output_root() / "s2_tumor_tiles" / "scores")
    s.add_argument("--lo", type=int, default=1)
    s.add_argument("--hi", type=int, default=None)
    s.add_argument("--overwrite", action="store_true")
    t = sub.add_parser("subset")
    t.add_argument("--scores-dir", type=Path, default=paths.output_root() / "s2_tumor_tiles" / "scores")
    t.add_argument("--conch-dir", type=Path, required=True)
    t.add_argument("--out-job-dir", type=Path, default=paths.output_root() / "s2_titan_tumor_only")
    t.add_argument("--coords-dir", default="40x_1024px_0px_overlap")
    t.add_argument("--patch-encoder", default="conch_v15")
    t.add_argument("--tile", type=int, default=512)
    t.add_argument("--min-tumour-tiles", type=int, default=2)
    t.add_argument("--min-patches", type=int, default=8)
    m = sub.add_parser("matrix")
    m.add_argument("--slide-features-dir", type=Path, required=True)
    m.add_argument("--model", default="titan")
    m.add_argument("--method", default="titan")
    m.add_argument("--metric", default="cosine")
    m.add_argument("--tumor-only", dest="tumor_only", action="store_true")
    m.add_argument("--min-slides", type=int, default=2500)
    g = sub.add_parser("filter")
    g.add_argument("--features-dir", type=Path, required=True,
                   help="source (features, coords) pickles, e.g. a staging dir of "
                        "symlinks to the CCDI query pkls")
    g.add_argument("--scores-dir", type=Path, required=True,
                   help="per-slide <slide>.csv.gz from a prior `score` run over "
                        "the SAME pickles")
    g.add_argument("--out-root", type=Path, required=True,
                   help="writes <out-root>/<arm>/<slide>.pkl + _filter_summary.csv")
    g.add_argument("--arms", default="passthrough,tumor_only,minus_artifacts")
    g.add_argument("--min-tiles", type=int, default=8,
                   help="warn-and-flag threshold, not a refusal (0 kept refuses)")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--skip-empty", dest="skip_empty", action="store_true",
                   help="slides whose arm keeps 0 tiles are skipped (recorded in the summary) "
                        "instead of aborting the run -- for filtering a whole library")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    rec = {"score": score, "subset": subset, "matrix": matrix,
           "filter": filter_features}[a.action](a)
    print(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
