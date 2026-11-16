"""C4 (R3.5) -- do the ROI similarity heatmaps land on annotated tumour?

The manuscript's ROI heatmap (Methods, "Visualization of the search results"):
for a query ROI and a reference slide, the per-mosaic-tile ROI distances are
projected onto the reference's 512-px grid (each mosaic tile's score is written
to every patch in its colour/space cluster, ``cluster_locs``; overlaps averaged),
min-max normalised within the query-reference pair and inverted. This module
computes exactly that map and scores it against the pathologist tile annotations
of the reference slide (``pathologist_annotated_labels``: patch_label 1 = Tumor,
2 = Immune cells, 3 = Necrosis, 4 = nonTumor, 5 = Artifacts, 0 = no annotation).

Design (pre-registered in plan v5 R3.5):

* references: annotated slides that are in the library (205), with at least
  ``--min-pos`` tumour-labelled patches;
* queries for a reference: ``--n-queries`` tumour-labelled mosaic tiles drawn
  (seeded) from *other* annotated slides of the **same tumour group**, different
  patient -- the situation the heatmap is meant for; and, as a control, the same
  number drawn from annotated slides of a **different group**;
* per (query, reference) pair: AUROC and average precision of the heatmap value
  for "tumour" over the reference's patches, the pointing game (is the
  highest-scoring patch tumour?), and the null AP = tumour fraction of the scored
  patches. Two ground-truth variants: **A** tumour vs explicitly annotated
  non-tumour (classes 2-5; only references that have >= ``--min-neg`` such
  patches), **B** tumour vs every other patch (unannotated counted as
  non-tumour -- noisier, but uses all 205 references);
* aggregation: mean per reference slide, then mean over references, with a
  patient-clustered bootstrap over reference patients (B, seed as the paper).

Distances are L1 between UNI2 mosaic-tile embeddings (the ROI pipeline's
metric). Nothing here uses the retrieval ranking; it tests the *map*, which is
what R3.5 asked about.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

TUMOR = 1
NONTUMOR = (2, 3, 4, 5)
COMET = Path("/PROSPER_WORKSPACE")
DEFAULT_FEATURES = COMET / "comet-feature-gen" / "comet-uni2" / "data" / "tmp_final"
DEFAULT_META = COMET / "comet-feature-gen" / "comet-uni2" / "data" / "input" / "mosaic512_1.3.1__metadata_0.5.pkl"
DEFAULT_LABELS = COMET / "comet-annotation" / "output" / "pathologist_annotated_labels"
DEFAULT_IDS = COMET / "comet-annotation" / "output" / "pathologist_annotated_slide_ids" / "slide_ids.txt"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


def _auroc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ap(score: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    hits = np.cumsum(ys)
    prec = hits / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


class Slide:
    """Mosaic features, per-patch label lookup and the cluster projection."""

    def __init__(self, sid: str, features_dir: Path, meta_tiles, labels_dir: Path):
        feats, coords = pd.read_pickle(features_dir / f"{sid}.pkl")
        self.sid = sid
        X = np.asarray(feats, dtype=np.float32)
        # align metadata tiles to feature rows by level-0 coordinate (the two
        # files agree in order for almost every slide, but not all -- 16973)
        pos = {tuple(c["wsi_loc"]): i for i, c in enumerate(coords)}
        keep = [(t, pos[tuple(t["wsi_loc"])]) for t in meta_tiles if tuple(t["wsi_loc"]) in pos]
        self.n_unmatched = len(meta_tiles) - len(keep)
        meta_tiles = [t for t, _ in keep]
        self.X = X[[i for _, i in keep]]
        coords = [{"wsi_loc": list(t["wsi_loc"])} for t in meta_tiles]
        lab = pd.read_csv(labels_dir / f"{sid}.csv", usecols=["wsi_loc_x", "wsi_loc_y", "patch_label"])
        self.label = {(int(x), int(y)): int(l) for x, y, l in
                      zip(lab.wsi_loc_x, lab.wsi_loc_y, lab.patch_label)}
        # patch grid: every cluster location of every mosaic tile
        locs, owner = [], []
        for j, t in enumerate(meta_tiles):
            for xy in t.get("cluster_locs", [t["wsi_loc"]]):
                locs.append((int(xy[0]), int(xy[1])))
                owner.append(j)
        self.patch_index = {}
        for p, j in zip(locs, owner):
            self.patch_index.setdefault(p, []).append(j)
        self.patches = list(self.patch_index)
        self.patch_label = np.array([self.label.get(p, 0) for p in self.patches], dtype=np.int8)
        self.tile_label = np.array([self.label.get(tuple(c["wsi_loc"]), 0) for c in coords], dtype=np.int8)
        # sparse projection: patch value = mean of its owning tiles' scores
        self._owners = [np.array(self.patch_index[p], dtype=np.int64) for p in self.patches]

    def heatmap(self, q: np.ndarray) -> np.ndarray:
        d = np.abs(self.X - q[None, :]).sum(axis=1)           # L1 per mosaic tile
        v = np.array([d[o].mean() for o in self._owners])
        lo, hi = v.min(), v.max()
        return 1.0 - (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def evaluate_pair(ref: Slide, q: np.ndarray, min_neg: int = 20) -> dict:
    h = ref.heatmap(q)
    y_t = (ref.patch_label == TUMOR)
    annotated = y_t | np.isin(ref.patch_label, NONTUMOR)
    out = {"n_patches": int(len(h)), "tumour_fraction_B": float(y_t.mean()),
           "pointing_B": bool(y_t[int(np.argmax(h))]),
           "auroc_B": _auroc(h, y_t.astype(int)), "ap_B": _ap(h, y_t.astype(int))}
    if annotated.sum() and y_t[annotated].sum() and (~y_t[annotated]).sum() >= min_neg:
        hA, yA = h[annotated], y_t[annotated].astype(int)
        out.update({"n_annotated": int(annotated.sum()), "tumour_fraction_A": float(yA.mean()),
                    "pointing_A": bool(yA[int(np.argmax(hA))]),
                    "auroc_A": _auroc(hA, yA), "ap_A": _ap(hA, yA)})
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--mosaic-metadata", type=Path, default=DEFAULT_META)
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--annotated-ids", type=Path, default=DEFAULT_IDS)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--roi-tiles", type=int, default=1,
                   help="query = mean embedding of this many tumour-labelled mosaic tiles of one "
                        "query slide (1 = a single tile; >1 approximates a query ROI)")
    p.add_argument("--min-pos", type=int, default=20)
    p.add_argument("--min-neg", type=int, default=20)
    p.add_argument("--limit-refs", type=int, default=None)
    p.add_argument("--iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--task", default="c4_localization")
    return p


def run(a) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    meta = pd.read_csv(paths.slide_metadata(), dtype=str)
    group = dict(zip(meta["Slide.Scan.File"], meta["Tumor.group"]))
    patient = dict(zip(meta["Slide.Scan.File"], meta["UID.Subject"]))
    ids = [l.strip() for l in open(a.annotated_ids) if l.strip()]
    in_lib = [s for s in ids if s in group and group[s] != "MISC"]
    raw = pd.read_pickle(a.mosaic_metadata)
    tiles = {e[0].split("/")[-1].replace(".svs", ""): e[1] for e in raw if e[1] is not None}
    slides = {}
    for s in in_lib:
        if s in tiles and (a.features_dir / f"{s}.pkl").exists():
            slides[s] = Slide(s, a.features_dir, tiles[s], a.labels_dir)
    print(f"[c4] {len(slides)} annotated in-library slides loaded ({time.time()-t0:.0f}s)", flush=True)
    # query pool: tumour-labelled mosaic tiles per slide
    pool = {s: np.where(sl.tile_label == TUMOR)[0] for s, sl in slides.items()}
    refs = [s for s, sl in slides.items() if (sl.patch_label == TUMOR).sum() >= a.min_pos]
    if a.limit_refs:
        refs = refs[:a.limit_refs]
    rows = []
    for r in refs:
        g, pt = group[r], patient[r]
        same = [s for s in slides if s != r and group[s] == g and patient[s] != pt and len(pool[s])]
        other = [s for s in slides if group[s] != g and group[s] != "NORMAL" and len(pool[s])]
        for arm, cand in (("same_group", same), ("other_group", other)):
            if not cand:
                continue
            for _ in range(a.n_queries):
                qs = cand[rng.integers(len(cand))]
                n_take = min(a.roi_tiles, len(pool[qs]))
                qidx = rng.choice(pool[qs], size=n_take, replace=False)
                qvec = slides[qs].X[qidx].mean(axis=0)
                res = evaluate_pair(slides[r], qvec, a.min_neg)
                rows.append({"reference": r, "ref_group": g, "ref_patient": pt, "arm": arm,
                             "query_slide": qs, "query_group": group[qs],
                             "query_tiles": ";".join(map(str, qidx.tolist())), **res})
    df = pd.DataFrame(rows)
    out_dir = paths.output_root() / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"per_pair_{a.task}.csv", index=False)

    metrics = ["auroc_A", "ap_A", "pointing_A", "tumour_fraction_A",
               "auroc_B", "ap_B", "pointing_B", "tumour_fraction_B"]

    def agg(d: pd.DataFrame) -> dict:
        per_ref = d.groupby("reference")[metrics].mean(numeric_only=True)
        return {m: float(per_ref[m].mean()) for m in metrics if m in per_ref}

    def boot(d: pd.DataFrame) -> dict:
        per_ref = d.groupby("reference")[metrics].mean(numeric_only=True)
        per_ref["patient"] = per_ref.index.map(patient)
        groups = {p: g.drop(columns="patient").to_numpy(dtype=float) for p, g in per_ref.groupby("patient")}
        keys = list(groups)
        cols = [m for m in metrics if m in per_ref.columns]
        acc = []
        brng = np.random.default_rng(a.seed)
        for _ in range(a.iter):
            take = brng.choice(len(keys), size=len(keys), replace=True)
            block = np.concatenate([groups[keys[i]] for i in take])
            acc.append(np.nanmean(block, axis=0))
        acc = np.array(acc)
        return {m: {"lo": float(np.nanpercentile(acc[:, i], 2.5)), "hi": float(np.nanpercentile(acc[:, i], 97.5))}
                for i, m in enumerate(cols)}

    summary = {}
    for arm, d in df.groupby("arm"):
        summary[arm] = {"n_pairs": int(len(d)), "n_references": int(d.reference.nunique()),
                        "n_references_variant_A": int(d.dropna(subset=["auroc_A"]).reference.nunique()),
                        "mean_over_references": agg(d), "ci": boot(d),
                        "per_group": {g: agg(dd) for g, dd in d.groupby("ref_group")}}
    # paired same-vs-other per reference
    piv = df.groupby(["reference", "arm"])[["auroc_B", "ap_B", "pointing_B"]].mean().unstack("arm")
    paired = {}
    if ("auroc_B", "same_group") in piv.columns and ("auroc_B", "other_group") in piv.columns:
        for m in ("auroc_B", "ap_B", "pointing_B"):
            dlt = (piv[(m, "same_group")] - piv[(m, "other_group")]).dropna()
            paired[m] = {"mean_delta": float(dlt.mean()), "n": int(len(dlt)),
                         "frac_positive": float((dlt > 0).mean())}
    res = {"config": {"features_dir": str(a.features_dir), "mosaic_metadata": str(a.mosaic_metadata),
                      "labels_dir": str(a.labels_dir), "n_queries": a.n_queries, "min_pos": a.min_pos,
                      "min_neg": a.min_neg, "roi_tiles": a.roi_tiles,
                      "metric": "L1 on UNI2 mosaic tile embeddings",
                      "bootstrap_B": a.iter, "seed": a.seed, "prosper_commit": _commit(),
                      "label_semantics": {"1": "Tumor", "2": "Immune cells", "3": "Necrosis",
                                          "4": "nonTumor", "5": "Artifacts", "0": "no annotation"}},
           "n_annotated_in_library": len(slides), "n_references": len(refs),
           "summary": summary, "same_minus_other_per_reference": paired,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(out_dir / f"{a.task}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    return res


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    res = run(a)
    for arm, s in res["summary"].items():
        m, ci = s["mean_over_references"], s["ci"]
        print(f"{arm}: refs {s['n_references']} (A: {s['n_references_variant_A']}) pairs {s['n_pairs']}")
        for k in ("auroc_B", "ap_B", "pointing_B", "tumour_fraction_B", "auroc_A", "ap_A", "pointing_A", "tumour_fraction_A"):
            if k in m:
                print(f"   {k:20s} {m[k]:.3f} [{ci[k]['lo']:.3f}, {ci[k]['hi']:.3f}]")
    print("same - other per reference:", res["same_minus_other_per_reference"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
