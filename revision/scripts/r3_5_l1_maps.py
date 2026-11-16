#!/usr/bin/env python3
"""R3.5-L1 -- localisation maps against pathologist annotation (specification 001, 2026-10-17).

Binding specification: $PROSPER_OUT/r3_5_localisation_maps_20261017/contract_001/SPECIFICATION.md
(+ amendments in later contract_* directories).  This script implements:

  bind      S0a  bind every input by sha256; extract the mosaic metadata (cluster_locs) of the
                 needed slides into a compact cache (the 2-GB metadata pickle takes ~75 s to load).
  f1        F1   ROI L1 map at the submitted setting (L1, mean aggregation) on the C4-ROI8 pairs of the
                 44 variant-A references: per-pair AUROC/AP/pointing/tumour share must equal C4 exactly.
  f2        F2   pipeline OT plan code (OTSlideEmbeddingGenerator.compute_ot_distance, POT exact EMD,
                 cosine cost, uniform marginals) reproduces the accepted all-tissue UNI2 OT distance for
                 5 pairs to 1e-6.
  f3        F3   clustering code against the published Fig. 5 partitions (saved plans; no new plan).
  pairs     S0b  reference sets, pair table (design P1 "natural retrieval pairs" or P2 "constructed
                 annotated pairs"), development/report split, F4/F5.     [HELD: specification amendment]
  s1-cmds / s1   per-reference OT plans and per-pair map scores for every grid setting. [HELD]
  s2 / s3        grid tables, selection, report-half bootstrap, outcome rule; montages.  [HELD]

Nothing here edits the dev repo; prosper-search code is imported read-only.  Every run writes a new
run_NNN directory (failed attempts are preserved) with COMPLETE.json or FAILED.json and sha256 bindings.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

H = Path("/PROSPER_WORKSPACE")
DEV = H / "prosper-search"
REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
ROOT = OUT / "r3_5_localisation_maps_20261017"
CONTRACT = ROOT / "contract_001" / "CONTRACT.md"

UNI2 = H / "comet-feature-gen" / "comet-uni2"
FEATURES = UNI2 / "data" / "tmp_final"
META_PKL = UNI2 / "data" / "input" / "mosaic512_1.3.1__metadata_0.5.pkl"
OT_MATRIX = UNI2 / "output" / "similarity_matrix" / "combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl"
SAVED_PLANS = UNI2 / "output" / "similarity_pairwise_ot"
LABELS = H / "comet-annotation" / "output" / "pathologist_annotated_labels"
POOL_DIR = OUT / "r2_1_filter_tf2_20261016" / "s0" / "run_002"
POOL = POOL_DIR / "pool.csv"
QUERY_ASSIGNMENT = POOL_DIR / "query_assignment.csv"
POOL_BINDINGS = POOL_DIR / "input_bindings.json"
EXCLUDED_20X = OUT / "r2_1_filter_tf2_20261016" / "contract_candidate_006" / "excluded_20x_slides.txt"
C4_CSV = OUT / "c4_localization_roi8" / "per_pair_c4_localization_roi8.csv"
C4_JSON = OUT / "c4_localization_roi8" / "c4_localization_roi8.json"
C4_REFS = OUT / "r2_5_c4_whole_patient_ci_20261008" / "per_reference_variant_A.csv"
C4_CI = OUT / "r2_5_c4_whole_patient_ci_20261008" / "c4_whole_patient_ci.json"
FIG5_DIR = OUT / "r3_5_fig5_ot_tile_montage_20261008_v2"
FIG5_REPLAY = OUT / "review_scratch_batch2_r25_20261006" / "lead_fig5_driver_match"
FIG5_PAIRS = [("A1", "16875", "15155", 4e-5), ("A2", "16875", "16867", 4e-5), ("B", "7333", "7330", 5e-4)]
CODE_FILES = {
    "localization_cli": DEV / "prosper" / "evaluation" / "localization_cli.py",
    "ot_visualization": DEV / "prosper" / "visualization" / "ot_visualization.py",
    "roi_heatmap": DEV / "prosper" / "visualization" / "roi_heatmap.py",
    "ot_embedding": DEV / "prosper" / "embeddings" / "slide_embedding" / "ot_embedding.py",
}

TUMOR = 1
NONTUMOR = (2, 3, 4, 5)
SEED = 123
N_BOOT = 5000
C4_MIN_NEG = 20
F2_TOL = 1e-6

# ---------------------------------------------------------------- parameter grid (specification section 3)
A_DISTANCES = ("l1", "l2", "cosine")
A_AGGREGATIONS = ("mean", "min")
KS = (1, 2, 3, 5, 8, 10)
QS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
LOUVAIN_RES_001 = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
# amendment 003: + {1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1} and the published 4e-5 (19 resolutions)
LOUVAIN_RES = (1e-5, 4e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1) + LOUVAIN_RES_001
LOUVAIN_TIMEOUT_S = 30


def grid_a():
    return [dict(family="A", setting=f"A|{d}|{g}", distance=d, aggregation=g)
            for d in A_DISTANCES for g in A_AGGREGATIONS]


def b_sparsifications():
    return [("none", None)] + [("mutual_topk", k) for k in KS]


def grid_b():
    out = []
    for stat in ("meancost", "mincost"):
        for sp, k in b_sparsifications():
            out.append(dict(family="B", setting=f"B|{stat}|{sp}|{k}", stat=stat, sparsify=sp, k=k))
    return out


def c_sparsifications():
    sp = [("none", None)]
    sp += [("topk", k) for k in KS]
    sp += [("mutual_topk", k) for k in KS]
    sp += [("threshold", q) for q in QS]
    return sp


def c_clusterings():
    return [("connected_components", None)] + [("louvain", r) for r in LOUVAIN_RES]


def grid_c():
    return [dict(family="C", setting=f"C|{sp}|{p}|{cl}|{r}", sparsify=sp, param=p, clustering=cl, resolution=r)
            for sp, p in c_sparsifications() for cl, r in c_clusterings()]




# ---------------------------------------------------------------- utilities
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def binding(path, role) -> dict:
    p = Path(path)
    st = p.stat()
    return {"role": role, "path": str(p), "resolved_path": str(p.resolve()), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(p)}


def new_run_dir(stage: str, root: Path = None) -> Path:
    base = (root or ROOT) / stage
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"run_{n:03d}").exists():
        n += 1
    d = base / f"run_{n:03d}"
    d.mkdir()
    return d


def latest_complete(stage: str, root: Path = None) -> Path:
    base = (root or ROOT) / stage
    runs = sorted(p for p in base.glob("run_*") if (p / "COMPLETE.json").exists())
    if not runs:
        raise RuntimeError(f"no COMPLETE run for stage {stage} under {base}")
    return runs[-1]


def verify_complete(run_dir: Path) -> dict:
    rec = json.loads((run_dir / "COMPLETE.json").read_text())
    for name, h in rec["outputs"].items():
        if sha256_file(run_dir / name) != h:
            raise RuntimeError(f"{run_dir}/{name}: sha256 differs from COMPLETE.json")
    return rec


def finish(run_dir: Path, stage: str, extra: dict) -> dict:
    outputs = {p.name: sha256_file(p) for p in sorted(run_dir.iterdir())
               if p.is_file() and p.name not in ("COMPLETE.json", "FAILED.json")}
    rec = {"stage": stage, "finished": now(), "script": str(Path(__file__).resolve()),
           "script_sha256": sha256_file(Path(__file__).resolve()), "host": platform.node(),
           "python": sys.version.split()[0], "outputs": outputs, **extra}
    (run_dir / "COMPLETE.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def fail(run_dir: Path, stage: str, reason: str, extra: dict = None):
    rec = {"stage": stage, "failed": now(), "reason": reason,
           "script_sha256": sha256_file(Path(__file__).resolve()), **(extra or {})}
    (run_dir / "FAILED.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"[{stage}] FAILED: {reason}", flush=True)
    return rec


def import_prosper():
    if str(DEV) not in sys.path:
        sys.path.insert(0, str(DEV))
    from prosper.evaluation import localization_cli as LC  # noqa: E402
    from prosper.visualization import ot_visualization as OV  # noqa: E402
    return LC, OV


def ot_generator():
    if str(DEV) not in sys.path:
        sys.path.insert(0, str(DEV))
    from prosper.embeddings.slide_embedding.ot_embedding import OTSlideEmbeddingGenerator
    return OTSlideEmbeddingGenerator(metric="cosine", reg=0.0, feature_dim=1536, n_workers=1,
                                     verbose=False, max_iter=1000000)


# ---------------------------------------------------------------- slides
def load_features(sid: str, features_dir: Path = FEATURES):
    feats, coords = pd.read_pickle(Path(features_dir) / f"{sid}.pkl")
    return np.asarray(feats, dtype=np.float32), [tuple(int(v) for v in c["wsi_loc"]) for c in coords]


def aligned_rows(coords, meta_tiles):
    """Feature-row index of each metadata tile, in metadata order, exactly as localization_cli.Slide."""
    pos = {tuple(c): i for i, c in enumerate(coords)}
    return [pos[tuple(t["wsi_loc"])] for t in meta_tiles if tuple(t["wsi_loc"]) in pos]


def load_labels(sid: str, labels_dir: Path = LABELS) -> dict:
    lab = pd.read_csv(Path(labels_dir) / f"{sid}.csv", usecols=["wsi_loc_x", "wsi_loc_y", "patch_label"])
    return {(int(x), int(y)): int(l) for x, y, l in zip(lab.wsi_loc_x, lab.wsi_loc_y, lab.patch_label)}


class MapSlide:
    """A reference/query slide: full OT feature rows, the localization_cli.Slide projection and labels."""

    def __init__(self, sid, meta_tiles, features_dir=FEATURES, labels_dir=LABELS, LC=None):
        if LC is None:
            LC, _ = import_prosper()
        self.sid = sid
        self.slide = LC.Slide(sid, Path(features_dir), meta_tiles, Path(labels_dir))   # imported projection
        self.F, self.coords = load_features(sid, features_dir)                    # all rows (OT input)
        self.rows = np.array(aligned_rows(self.coords, meta_tiles), dtype=np.int64)  # Slide.X row -> F row
        if not np.array_equal(self.F[self.rows], self.slide.X):
            raise RuntimeError(f"{sid}: feature alignment differs from localization_cli.Slide")
        lab = self.slide.label
        self.row_label = np.array([lab.get(c, 0) for c in self.coords], dtype=np.int8)   # per OT row
        self.owners = self.slide._owners
        self.patch_label = self.slide.patch_label


# ---------------------------------------------------------------- maps
def project(tile_values: np.ndarray, owners, aggregation: str) -> np.ndarray:
    """Tile score -> patch score over cluster_locs (localization_cli projection; 'min' = min over owners)."""
    if aggregation == "mean":
        return np.array([tile_values[o].mean() for o in owners])
    if aggregation == "min":
        return np.array([tile_values[o].min() for o in owners])
    raise ValueError(aggregation)


def normalise_invert(v: np.ndarray) -> np.ndarray:
    """Exactly localization_cli.Slide.heatmap's min-max normalisation and inversion."""
    lo, hi = v.min(), v.max()
    return 1.0 - (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def roi_distances(X: np.ndarray, q: np.ndarray, distance: str) -> np.ndarray:
    if distance == "l1":
        return np.abs(X - q[None, :]).sum(axis=1)          # identical expression to Slide.heatmap
    if distance == "l2":
        return np.sqrt(((X - q[None, :]) ** 2).sum(axis=1))
    if distance == "cosine":
        num = X @ q
        den = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
        return 1.0 - num / den
    raise ValueError(distance)


def roi_map(slide, q: np.ndarray, distance: str, aggregation: str) -> np.ndarray:
    """ROI heatmap on the reference patch grid (higher = more similar)."""
    d = roi_distances(slide.X, q, distance)
    return normalise_invert(project(d, slide._owners, aggregation))


def score_continuous(h: np.ndarray, patch_label: np.ndarray, LC, min_neg: int = C4_MIN_NEG) -> dict:
    """Variant-A metrics as localization_cli.evaluate_pair (A branch), using its _auroc/_ap."""
    y_t = (patch_label == TUMOR)
    annotated = y_t | np.isin(patch_label, NONTUMOR)
    out = {}
    if annotated.sum() and y_t[annotated].sum() and (~y_t[annotated]).sum() >= min_neg:
        hA, yA = h[annotated], y_t[annotated].astype(int)
        out = {"n_annotated": int(annotated.sum()), "tumour_fraction_A": float(yA.mean()),
               "pointing_A": bool(yA[int(np.argmax(hA))]),
               "auroc_A": LC._auroc(hA, yA), "ap_A": LC._ap(hA, yA)}
    return out


# ---------------------------------------------------------------- OT
def ot_plan(gen, Fq: np.ndarray, Fr: np.ndarray):
    """Pipeline OT (query rows x reference columns). Returns (T, M, distance)."""
    res = gen.compute_ot_distance(Fq, Fr, return_transport=True)
    return np.asarray(res["T"], dtype=np.float64), np.asarray(res["M"]), float(res["distance"])


def threshold_sparsify(T: np.ndarray, q: float) -> np.ndarray:
    """Mass-quantile threshold: generalises ot_visualization's 'threshold' branch (fixed 90th percentile of the
    nonzero entries) to quantile q; identical to it at q = 0.90 (tested)."""
    threshold = np.percentile(T[T > 0], 100.0 * q) if np.any(T > 0) else 0
    return np.where(T >= threshold, T, 0.0)


def sparsify(T: np.ndarray, method: str, param, OV) -> np.ndarray:
    if method == "none":
        return T
    if method in ("topk", "mutual_topk"):
        return OV.sparsify_transport_matrix(T, k=int(param), method=method)
    if method == "threshold":
        return threshold_sparsify(T, float(param))
    raise ValueError(method)


def cluster_sparse(T_sp: np.ndarray, method: str, resolution) -> tuple:
    """Graph build and clustering exactly as ot_visualization.cluster_patches_from_transport, but on an
    already-sparsified plan (that function hard-codes mutual top-k).  Returns (lab_query, lab_ref), 1-based."""
    import networkx as nx
    n_a, n_b = T_sp.shape
    G = nx.Graph()
    G.add_nodes_from(range(n_a), bipartite=0, slide='A')
    G.add_nodes_from(range(n_a, n_a + n_b), bipartite=1, slide='B')
    i_idx, j_idx = np.nonzero(T_sp)
    for i, j in zip(i_idx, j_idx):
        G.add_edge(i, n_a + j, weight=float(T_sp[i, j]))
    partition = {}
    if method == "connected_components":
        for cid, comp in enumerate(nx.connected_components(G), start=1):
            for u in comp:
                partition[u] = cid
    elif method == "louvain":
        from networkx.algorithms.community import louvain_communities
        for cid, nodes in enumerate(louvain_communities(G, weight="weight", resolution=resolution, seed=0), start=1):
            for u in nodes:
                partition[u] = cid
    else:
        raise ValueError(method)
    lab = np.array([partition[u] for u in range(n_a + n_b)], dtype=np.int64)
    return lab[:n_a], lab[n_a:]


# ---------------------------------------------------------------- pair designs (amendment 002 section 2)
def numeric_sorted(ids):
    return sorted((str(s) for s in ids), key=lambda s: (len(s), s))


def eligible_references(patch_counts: pd.DataFrame, min_pos: int, min_neg: int) -> list:
    """patch_counts: slide_id, n_tumour_patches, n_nontumour_patches (variant-A patch grid = cluster locations)."""
    ok = patch_counts[(patch_counts.n_tumour_patches >= min_pos) & (patch_counts.n_nontumour_patches >= min_neg)]
    return numeric_sorted(ok.slide_id)


def population_topk(values: np.ndarray, ids, patients, k: int = 10) -> dict:
    """Top-k neighbours of every row of a square distance matrix restricted to the retrieval population:
    ascending distance, self, same-patient (by patient id) and masked (>= 1e5, the stored 1e6) entries
    excluded, stable order (column order = population order).  Returns {query: [(neighbour, rank), ...]}."""
    ids = list(ids)
    pat = np.asarray(patients)
    out = {}
    for i, q in enumerate(ids):
        sc = values[i].astype(float).copy()
        bad = (pat == pat[i]) | (sc >= 1e5) | ~np.isfinite(sc)
        bad[i] = True
        sc[bad] = np.inf
        order = np.argsort(sc, kind="stable")[:k]
        out[q] = [(ids[j], r + 1) for r, j in enumerate(order) if np.isfinite(sc[j])]
    return out


def independent_topk(sub: pd.DataFrame, query: str, patient_of: dict, k: int = 10) -> list:
    """F6 re-derivation by a different route (pandas Series filtering + nsmallest)."""
    s = sub.loc[query].drop(index=query)
    s = s[[patient_of[c] != patient_of[query] for c in s.index]]
    s = s[s < 1e5]
    return list(s.nsmallest(k, keep="first").index)


def pairs_p1(refs, ref_info: pd.DataFrame, population: pd.DataFrame, topk: dict) -> pd.DataFrame:
    """P1 natural retrieval pairs: every population query slide of another patient whose top-10 contains the
    reference.  ref_info / population: slide_id, patient, group."""
    ref_info = ref_info.set_index("slide_id")
    pop = population.set_index("slide_id")
    refset = set(refs)
    rows = []
    for q in numeric_sorted(pop.index):
        for nb, rank in topk.get(q, []):
            if nb not in refset:
                continue
            pt, g = ref_info.at[nb, "patient"], ref_info.at[nb, "group"]
            qp, qg = pop.at[q, "patient"], pop.at[q, "group"]
            if qp == pt or q == nb:
                continue
            rows.append({"design": "P1", "reference": nb, "ref_patient": pt, "ref_group": g,
                         "arm": "same_group" if qg == g else "other_group", "query": q, "query_patient": qp,
                         "query_group": qg, "rank_of_reference_in_query_list": rank})
    df = pd.DataFrame(rows, columns=["design", "reference", "ref_patient", "ref_group", "arm", "query",
                                     "query_patient", "query_group", "rank_of_reference_in_query_list"])
    if len(df):
        df["_r"] = df.reference.map(lambda s: (len(s), s))
        df["_q"] = df["query"].map(lambda s: (len(s), s))
        df = df.sort_values(["_r", "_q"]).drop(columns=["_r", "_q"]).reset_index(drop=True)
    return df


def draw_rois(query_ids, labelled_rows: dict, predicted_rows: dict, roi_tiles: int = 8, seed: int = SEED) -> dict:
    """Per query slide (numeric order, one PCG64(seed) stream): up to roi_tiles tumour-labelled feature rows
    (if any), then up to roi_tiles TF-2-predicted-tumour feature rows (if any), without replacement."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = {}
    for q in numeric_sorted(set(query_ids)):
        rec = {}
        for kind, src in (("labelled", labelled_rows), ("predicted", predicted_rows)):
            pool = np.asarray(src.get(q, ()), dtype=np.int64)
            if len(pool):
                rec[kind] = np.sort(rng.choice(pool, size=min(roi_tiles, len(pool)), replace=False)).tolist()
        out[q] = rec
    return out


def c4_draw(slide_order, tile_tumour: dict, patch_tumour_count: dict, group: dict, patient: dict,
            n_queries: int = 10, roi_tiles: int = 8, min_pos: int = 20, seed: int = SEED) -> pd.DataFrame:
    """Verbatim re-implementation of the pair draw in localization_cli.run (C4): references = slides with
    >= min_pos tumour patches in slide order; per reference and arm, n_queries query slides drawn WITH
    replacement (rng.integers) from same-group other-patient / other-group non-NORMAL candidates having >= 1
    tumour-labelled mosaic tile; query tiles rng.choice(pool, min(roi_tiles, n), replace=False)."""
    rng = np.random.default_rng(seed)
    pool = {s: np.asarray(tile_tumour[s]) for s in slide_order}
    refs = [s for s in slide_order if patch_tumour_count[s] >= min_pos]
    rows = []
    for r in refs:
        g, pt = group[r], patient[r]
        same = [s for s in slide_order if s != r and group[s] == g and patient[s] != pt and len(pool[s])]
        other = [s for s in slide_order if group[s] != g and group[s] != "NORMAL" and len(pool[s])]
        for arm, cand in (("same_group", same), ("other_group", other)):
            if not cand:
                continue
            for _ in range(n_queries):
                qs = cand[rng.integers(len(cand))]
                n_take = min(roi_tiles, len(pool[qs]))
                qidx = rng.choice(pool[qs], size=n_take, replace=False)
                rows.append({"reference": r, "ref_group": g, "ref_patient": pt, "arm": arm, "query": qs,
                             "query_group": group[qs], "query_patient": patient[qs],
                             "query_tiles": ";".join(map(str, qidx.tolist()))})
    return pd.DataFrame(rows)


def split_patients(ref_patient_group: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Development/report split of reference patients by seeded hash, stratified by group.
    Within each group (sorted by name), patients are ordered by sha256(f"{seed}|{patient}") and assigned
    alternately; the odd patient of successive odd-sized groups alternates between halves."""
    df = ref_patient_group.drop_duplicates("patient").copy()
    out, odd_counter = [], 0
    for g in sorted(df.group.unique()):
        ps = sorted(df[df.group == g].patient, key=lambda p: hashlib.sha256(f"{seed}|{p}".encode()).hexdigest())
        start = 0
        if len(ps) % 2 == 1:
            start = odd_counter % 2
            odd_counter += 1
        for i, p in enumerate(ps):
            out.append({"patient": p, "group": g, "half": "development" if (i + start) % 2 == 0 else "report"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- cluster metrics (specification 4)
def class_of(labels: np.ndarray, binary: bool) -> np.ndarray:
    """0 = unannotated; binary: 1 tumour, 2 non-tumour; 5-class: the label itself."""
    labels = np.asarray(labels)
    if not binary:
        return labels.copy()
    return np.where(labels == TUMOR, 1, np.where(np.isin(labels, NONTUMOR), 2, 0))


def cross_slide_agreement(T_sp, lab_q, lab_r, binary=True) -> dict:
    cq, cr = class_of(lab_q, binary), class_of(lab_r, binary)
    i, j = np.nonzero(T_sp)
    w = T_sp[i, j]
    ann = (cq[i] > 0) & (cr[j] > 0)
    mass = float(w[ann].sum())
    if mass <= 0:
        return {"agreement": float("nan"), "chance": float("nan"), "annotated_mass": 0.0}
    agree = float(w[ann & (cq[i] == cr[j])].sum()) / mass
    classes = sorted(set(cq[cq > 0]) | set(cr[cr > 0]))
    pq = {c: float((cq == c).sum()) / max(1, (cq > 0).sum()) for c in classes}
    pr = {c: float((cr == c).sum()) / max(1, (cr > 0).sum()) for c in classes}
    chance = float(sum(pq[c] * pr[c] for c in classes))
    return {"agreement": agree, "chance": chance, "annotated_mass": mass}


def cluster_stats(T_sp, cl_q, cl_r, lab_q, lab_r, binary=True) -> dict:
    from sklearn.metrics import normalized_mutual_info_score
    cq, cr = class_of(lab_q, binary), class_of(lab_r, binary)
    ids = np.concatenate([cl_q, cl_r])
    cls = np.concatenate([cq, cr])
    i, j = np.nonzero(T_sp)
    w = T_sp[i, j]
    intra = cl_q[i] == cl_r[j]
    cmass = pd.Series(w[intra]).groupby(cl_q[i][intra]).sum()
    sizes = pd.Series(ids).value_counts()
    num = den = 0.0
    for c, m in cmass.items():
        lab = cls[(ids == c) & (cls > 0)]
        if len(lab) and m > 0:
            num += m * np.bincount(lab).max() / len(lab)
            den += m
    ref_ann = cr > 0
    nmi = (float(normalized_mutual_info_score(cr[ref_ann], cl_r[ref_ann])) if ref_ann.sum() >= 2 else float("nan"))
    return {"purity": num / den if den > 0 else float("nan"), "n_clusters": int(len(sizes)),
            "median_cluster_size": float(np.median(sizes.values)),
            "frac_singleton_clusters": float((sizes.values == 1).mean()),
            "nmi_ref": nmi}


# ---------------------------------------------------------------- bootstrap (as r2_5_c4_whole_patient_ci.py)
def make_patient_draws(patient_ids, count=N_BOOT, seed=SEED):
    patients = tuple(sorted(set(patient_ids)))
    p = len(patients)
    mult = np.zeros((count, p), dtype=np.int64)
    if p:
        rng = np.random.Generator(np.random.PCG64(seed))
        for b in range(count):
            mult[b] = np.bincount(rng.integers(p, size=p), minlength=p)
    return patients, mult


def whole_patient_ci(values, patients, count=N_BOOT, seed=SEED) -> dict:
    """Mean over references; references weighted by their patient's bootstrap multiplicity."""
    v = np.asarray(values, float)
    pats, mult = make_patient_draws(list(patients), count, seed)
    W = mult[:, pd.Index(pats).get_indexer(list(patients))].astype(float)
    reps = (W @ v) / W.sum(1)
    return {"point": float(v.mean()), "ci95_lo": float(np.percentile(reps, 2.5)),
            "ci95_hi": float(np.percentile(reps, 97.5)), "n_references": int(len(v)),
            "n_patients": int(len(pats)), "B": int(count), "seed": int(seed)}


# ---------------------------------------------------------------- stages
def meta_cache_load(path=None) -> dict:
    path = path or (latest_complete("s0_bind") / "meta_tiles.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


C4_IDS = H / "comet-annotation" / "output" / "pathologist_annotated_slide_ids" / "slide_ids.txt"
C4_META = H / "comet-feature-gen" / "comet-analysis" / "data" / "final_scan_id_disease_map_with_sample_id_patient_id.csv"
TF2_DIR = OUT / "r2_1_filter_tf2_20261016" / "s3" / "uni2" / "b100"
WSI = H / "data" / "comet_final_HPC" / "{}.svs"
AMENDMENT_002 = ROOT / "contract_002" / "AMENDMENT_002_PAIRS.md"

SUBMITTED = {"A": "A|l1|mean",                         # Fig. 6B ROI heatmap as in Methods / C4
             "B": "B|meancost|none|None",               # ot_visualization weighted-distance overlay (sum_i T_ij M_ij,
                                                        #   = mean cost x 1/n_r under uniform marginals, unsparsified)
             "C": "C|none|None|louvain|4e-05"}          # amendment 003: published recipe (full plan); 4e-5 = the
                                                        #   published RMS value, the reference setting for new pairs

IMPLEMENTATION_CHOICES = """# R3.5-L1 implementation choices (pre-registered by the implementer before any map is scored)

Written by scripts/r3_5_l1_maps.py stage `pairs` into the S0b run directory and bound by sha256 in its
COMPLETE.json, i.e. before S1 computes any map score. Each item resolves a point the contract (001 +
amendment 002) does not fix; each is flagged for lead ratification in IMPLEMENTER_REPORT.md.

1. Submitted settings. A = L1, mean aggregation (C4). B = mass-weighted mean cost, no sparsification: the
   shipped `compute_per_patch_weighted_distance_scores` gives D_B = sum_i T_ij M_ij, which under uniform
   marginals and no sparsification equals the mean cost divided by n_ref, so it ranks tiles identically.
   C = mutual top-5 + Louvain resolution 1.0, as contract 001 s3 labels it ("1.0 (submitted default)") with the
   code default k = 5. NOTE: the published Fig. 5 used the full (unsparsified) plan with Louvain resolution
   5e-4 (RB) / 4e-5 (RMS) (thin-reset plan s12.10; F3 reports it); that recipe is NOT in the contract grid and is
   not added here.
2. Family B, reference tiles left without any link after sparsification: given the coldest value of the pair
   (cost = max linked value + 1), so they rank below every linked tile; coverage (share of reference tiles
   and of annotated patches with >= 1 linked owning tile) is recorded per pair and setting.
3. Family B projection onto patches: mean over owning tiles (the submitted aggregation), then min-max
   normalisation and inversion exactly as localization_cli.Slide.heatmap.
4. Family C mass-quantile threshold q: keep plan entries >= the q-th percentile (numpy linear) of the nonzero
   entries -- ot_visualization's 'threshold' branch with its fixed 90th percentile generalised to q (identical
   at q = 0.90). Top-k / mutual top-k: ot_visualization.sparsify_transport_matrix (rows = query tiles).
   Graph and clustering: the cluster_patches_from_transport construction (integer nodes, all tiles added as
   nodes, edges = nonzero sparsified entries weighted by mass; connected components or networkx
   louvain_communities seed 0), applied to the already-sparsified plan.
5. Family C metrics use the mosaic-tile labels at each tile's wsi_loc (OT rows of both slides); 'annotated' =
   label 1-5. Cross-slide class agreement = share of sparsified mass on links joining two annotated tiles that
   joins tiles of the same class; chance = sum_c p_query(c) p_ref(c) over the annotated-tile class shares
   (tile counts = mass shares under uniform marginals). Primary binary (tumour vs classes 2-5), secondary
   5-class. Purity: clusters weighted by their intra-cluster sparsified mass; clusters without an annotated
   tile or without intra mass are skipped. NMI: sklearn normalized_mutual_info_score (arithmetic) between
   cluster id and class over the annotated reference tiles.
6. Degeneracy: a pair is degenerate under a setting if it yields one cluster or more than half of its
   clusters are singletons; a setting is degenerate if > 50 % of the development P2 same-group pairs are.
7. Family C selection: the criterion (development P2 same-group mean of agreement - chance, per reference
   then over references) does not depend on the clustering, so the ten clusterings of one sparsification tie
   exactly. Ties are broken by the higher development P2 same-group binary purity, then by grid order.
   Families A/B: development P1 same-group mean AUROC (variant A; A uses the primary ROI); ties by grid order.
8. Outcome rule per family: adopt the selected setting iff selected != submitted and the report-half paired
   difference (selected - submitted, per reference, whole-patient bootstrap B = 5,000, PCG64 seed 123,
   percentile 95 %) has its lower bound > 0. A/B are judged on P1 same-group AUROC, C on P2 same-group
   agreement - chance.
9. P1: retrieval population = query_assignment.csv (2,604 slides); neighbour universe = the same 2,604 slides
   (the bound all-tissue matrix restricted to population rows and columns), self/same-patient (patient_id)/
   1e6-masked entries excluded, ascending distance, stable order. References outside the population
   (NORMAL/MISC) have no P1 pairs.
10. P1 query ROI: one PCG64(123) stream over the P1 query slides in numeric order; per slide up to 8 tumour-
    labelled feature rows (pathologist label 1 at the row's wsi_loc; slide 'annotated' if it has >= 1) and up to 8
    rows with TF-2 UNI2 full-head mask True (b100 full_mask.npy, rows aligned by slide_offsets and checked
    against the feature coordinates); primary ROI = labelled if available, else predicted; both scored.
11. P2: the C4 construction re-run verbatim (localization_cli.run draw loop, its 196-slide set and
    order, comet-analysis groups/patients, seed 123, draws with replacement) and required to equal the C4
    per-pair CSV; references in it take their C4 rows. Other references (not in C4's reference list) take
    the same loop as a continuation: one default_rng(123) stream over those references in numeric order,
    candidates = the 202 annotated library slides (pool groups/patients), other-group excluding NORMAL,
    query >= 1 tumour-labelled mosaic tile.
12. Split: reference patients of the strict (>= 20/20) set are split first (split_patients: per group,
    sha256("123|patient") order, alternating; odd groups alternate the extra patient), then the
    sensitivity-only patients the same way. Same split for P1 and P2.
13. Representative tiles (S3): pairs = the three Fig. 5 pairs + three P2 same-group pairs with distinct
    references drawn by PCG64(123); medoid link = intra-cluster link (i, j) minimising
    ||x_i - mu|| + ||y_j - mu|| with x, y the L2-normalised UNI2 vectors and mu their mean over all tiles of the
    cluster (both slides); highest- and lowest-mass intra-cluster links; ties by plan order. Tiles read at level
    0 (512 px at wsi_loc) from the HPC WSI alias.
"""


def stage_bind(a) -> int:
    d = new_run_dir("s0_bind")
    try:
        files = [binding(CONTRACT, "contract"), binding(AMENDMENT_002, "amendment_002"), binding(POOL, "pool"),
                 binding(QUERY_ASSIGNMENT, "query_assignment"), binding(POOL_BINDINGS, "pool_input_bindings"),
                 binding(EXCLUDED_20X, "excluded_20x"), binding(META_PKL, "mosaic_metadata"),
                 binding(OT_MATRIX, "ot_matrix_all_tissue"), binding(C4_CSV, "c4_roi8_per_pair"),
                 binding(C4_JSON, "c4_roi8_json"), binding(C4_REFS, "c4_whole_patient_per_reference"),
                 binding(C4_CI, "c4_whole_patient_ci"), binding(C4_IDS, "c4_annotated_ids"),
                 binding(C4_META, "c4_slide_metadata")]
        for f in ("full_mask.npy", "full_predictions.npy", "coordinates.npy", "slide_offsets.csv", "COMPLETE.json"):
            files.append(binding(TF2_DIR / f, f"tf2_uni2_b100:{f}"))
        files += [binding(p, f"code:{k}") for k, p in CODE_FILES.items()]
        for _, q, n, _r in FIG5_PAIRS:
            files.append(binding(SAVED_PLANS / f"ot_{q}_vs_{n}.pkl", f"fig5_plan:{q}_{n}"))
            files.append(binding(FIG5_DIR / f"labels_{q}_{n}.csv", f"fig5_labels:{q}_{n}"))
            files.append(binding(FIG5_REPLAY / f"replay_{q}_{n}_{'0.0005' if q == '7333' else '4e-05'}.json",
                                 f"fig5_replay:{q}_{n}"))
        lib_map = json.loads(POOL_BINDINGS.read_text())
        lm = [f for f in lib_map["files"] if f["role"] == "library_map"][0]
        files.append(binding(lm["path"], "library_map"))
        pool = pd.read_csv(POOL, dtype=str)
        ann = pool[(pool.in_pool == "1") & (pool.in_final_library == "1")]
        c4 = pd.read_csv(C4_CSV, dtype=str)
        c4ids = [l.strip() for l in open(C4_IDS) if l.strip()]
        need = (set(ann.slide_id) | set(c4.reference) | set(c4.query_slide) | set(c4ids)
                | {s for _, q, n, _r in FIG5_PAIRS for s in (q, n)})
        per_slide = []
        for s in numeric_sorted(need):
            rec = {"slide_id": s, "in_annotated_library_202": s in set(ann.slide_id)}
            for kind, p in (("features", FEATURES / f"{s}.pkl"), ("labels", LABELS / f"{s}.csv")):
                rec[f"{kind}_sha256"] = sha256_file(p) if p.exists() else None
            per_slide.append(rec)
        t0 = time.time()
        raw = pd.read_pickle(META_PKL)
        tiles = {e[0].split("/")[-1].replace(".svs", ""): e[1] for e in raw if e[1] is not None}
        del raw
        cache = {s: [{"wsi_loc": list(t["wsi_loc"]),
                      "cluster_locs": [list(x) for x in t.get("cluster_locs", [t["wsi_loc"]])]} for t in tiles[s]]
                 for s in need if s in tiles}
        with open(d / "meta_tiles.pkl", "wb") as f:
            pickle.dump(cache, f, protocol=4)
        (d / "meta_keys.json").write_text(json.dumps(numeric_sorted(tiles)))
        pd.DataFrame(per_slide).to_csv(d / "per_slide_bindings.csv", index=False)
        missing_meta = numeric_sorted(need - set(cache))
        (d / "input_bindings.json").write_text(json.dumps({"created": now(), "files": files, "n_slides": len(need),
                                                           "missing_metadata": missing_meta,
                                                           "meta_load_s": round(time.time() - t0, 1)}, indent=1))
        finish(d, "s0_bind", {"n_slides": len(need), "n_annotated_library": int(len(ann)),
                              "missing_metadata": missing_meta})
        print(f"[bind] {d}: {len(need)} slides, metadata missing {missing_meta}", flush=True)
        return 0
    except Exception as e:  # pragma: no cover
        fail(d, "s0_bind", repr(e), {"traceback": traceback.format_exc()})
        return 2


def bound_path(bind_rec_dir: Path, role: str) -> Path:
    b = json.loads((bind_rec_dir / "input_bindings.json").read_text())
    f = [x for x in b["files"] if x["role"] == role]
    if len(f) != 1:
        raise RuntimeError(f"binding role {role}: {len(f)} entries")
    p = Path(f[0]["path"])
    if sha256_file(p) != f[0]["sha256"]:
        raise RuntimeError(f"{p}: sha256 differs from the S0 binding")
    return p


def tf2_predicted_rows(bind: Path, slide_ids) -> tuple:
    off = pd.read_csv(bound_path(bind, "tf2_uni2_b100:slide_offsets.csv"), dtype={"slide_id": str})
    off = off.set_index("slide_id")
    mask = np.load(bound_path(bind, "tf2_uni2_b100:full_mask.npy"), mmap_mode="r")
    pred = np.load(bound_path(bind, "tf2_uni2_b100:full_predictions.npy"), mmap_mode="r")
    coords = np.load(bound_path(bind, "tf2_uni2_b100:coordinates.npy"), mmap_mode="r")
    out, checks = {}, []
    for s in slide_ids:
        a0, a1 = int(off.at[s, "tile_start"]), int(off.at[s, "tile_stop"])
        _, fc = load_features(s)
        ok = len(fc) == a1 - a0 and np.array_equal(np.asarray(coords[a0:a1]), np.asarray(fc, dtype=np.int64))
        if not ok:
            raise RuntimeError(f"{s}: TF-2 coordinates differ from the feature rows")
        m = np.asarray(mask[a0:a1])
        out[s] = np.where(m)[0].tolist()
        checks.append({"slide_id": s, "rows": a1 - a0, "mask_true": int(m.sum()),
                       "mask_equals_pred_tumour": bool(np.array_equal(m, np.asarray(pred[a0:a1]) == TUMOR))})
    return out, pd.DataFrame(checks)


def labelled_tumour_rows(s: str) -> list:
    p = LABELS / f"{s}.csv"
    if not p.exists():
        return []
    lab = load_labels(s)
    _, fc = load_features(s)
    return [i for i, c in enumerate(fc) if lab.get(c, 0) == TUMOR]


def stage_pairs(a) -> int:
    d = new_run_dir("s0_pairs")
    try:
        bind = latest_complete("s0_bind")
        verify_complete(bind)
        (d / "IMPLEMENTATION_CHOICES.md").write_text(IMPLEMENTATION_CHOICES)
        LC, _ = import_prosper()
        meta = meta_cache_load(bind / "meta_tiles.pkl")
        meta_keys = set(json.loads((bind / "meta_keys.json").read_text()))
        pool = pd.read_csv(bound_path(bind, "pool"), dtype=str)
        ann = pool[(pool.in_pool == "1") & (pool.in_final_library == "1")].copy()
        ann_ids = numeric_sorted(ann.slide_id)
        pat = dict(zip(ann.slide_id, ann["UID.Subject"]))
        grp = dict(zip(ann.slide_id, ann.diagnostic_group))
        cm = pd.read_csv(bound_path(bind, "c4_slide_metadata"), dtype=str)
        g4 = dict(zip(cm["Slide.Scan.File"], cm["Tumor.group"]))
        p4 = dict(zip(cm["Slide.Scan.File"], cm["UID.Subject"]))
        ids = [l.strip() for l in open(bound_path(bind, "c4_annotated_ids")) if l.strip()]
        in_lib = [s for s in ids if s in g4 and g4[s] != "MISC"]
        c4_order = [s for s in in_lib if s in meta_keys and (FEATURES / f"{s}.pkl").exists()]
        slides = {}
        for s in numeric_sorted(set(ann_ids) | set(c4_order)):
            slides[s] = LC.Slide(s, FEATURES, meta[s], LABELS)
        tile_tumour = {s: np.where(sl.tile_label == TUMOR)[0] for s, sl in slides.items()}
        n_tum = {s: int((sl.patch_label == TUMOR).sum()) for s, sl in slides.items()}
        n_non = {s: int(np.isin(sl.patch_label, NONTUMOR).sum()) for s, sl in slides.items()}
        checks = {}
        # ---- P2 part 1: verbatim C4 draw, must equal the C4 per-pair CSV
        c4rep = c4_draw(c4_order, tile_tumour, n_tum, g4, p4)
        c4csv = pd.read_csv(bound_path(bind, "c4_roi8_per_pair"), dtype=str)
        cmp_cols = [("reference", "reference"), ("arm", "arm"), ("query", "query_slide"),
                    ("query_tiles", "query_tiles"), ("ref_group", "ref_group"), ("query_group", "query_group")]
        eq = len(c4rep) == len(c4csv) and all((c4rep[x].astype(str).values == c4csv[y].values).all() for x, y in cmp_cols)
        checks["P2_c4_draw_equals_c4_csv"] = bool(eq)
        checks["c4_slide_set"] = len(c4_order)
        if not eq:
            (d / "checks.json").write_text(json.dumps(checks, indent=1))
            fail(d, "s0_pairs", "C4 draw replication differs from the C4 per-pair CSV (F1/P2 identity)", checks)
            return 3
        # ---- references
        pc = pd.DataFrame({"slide_id": ann_ids, "n_tumour_patches": [n_tum[s] for s in ann_ids],
                           "n_nontumour_patches": [n_non[s] for s in ann_ids]})
        strict = eligible_references(pc, 20, 20)
        loose = eligible_references(pc, 10, 10)
        assert set(strict) <= set(loose)
        pc["patient"] = pc.slide_id.map(pat)
        pc["group"] = pc.slide_id.map(grp)
        pc["strict"] = pc.slide_id.isin(strict)
        pc["loose"] = pc.slide_id.isin(loose)
        pc.to_csv(d / "reference_candidates.csv", index=False)
        # ---- P2
        c4refs = set(c4rep.reference)
        p2_parts = []
        from_c4 = [r for r in loose if r in c4refs]
        cont = [r for r in loose if r not in c4refs]
        x = c4rep[c4rep.reference.isin(from_c4)].copy()
        x["p2_source"] = "c4_draw"
        p2_parts.append(x)
        if cont:
            rng = np.random.default_rng(SEED)
            rows = []
            for r in cont:
                g, pt = grp[r], pat[r]
                same = [s for s in ann_ids if s != r and grp[s] == g and pat[s] != pt and len(tile_tumour[s])]
                other = [s for s in ann_ids if grp[s] != g and grp[s] != "NORMAL" and len(tile_tumour[s])]
                for arm, cand in (("same_group", same), ("other_group", other)):
                    if not cand:
                        continue
                    for _ in range(10):
                        qs = cand[rng.integers(len(cand))]
                        qidx = rng.choice(tile_tumour[qs], size=min(8, len(tile_tumour[qs])), replace=False)
                        rows.append({"reference": r, "ref_group": g, "ref_patient": pt, "arm": arm, "query": qs,
                                     "query_group": grp[qs], "query_patient": pat[qs],
                                     "query_tiles": ";".join(map(str, qidx.tolist())), "p2_source": "continuation"})
            p2_parts.append(pd.DataFrame(rows))
        p2 = pd.concat(p2_parts, ignore_index=True)
        p2["design"] = "P2"
        # pool group/patient for every P2 slide (C4 used the comet-analysis map) -- must agree
        for col, ref_col, m in (("ref_patient", "reference", pat), ("query_patient", "query", pat)):
            mism = int((p2[col] != p2[ref_col].map(m)).sum())
            checks[f"P2_{col}_c4map_vs_pool_mismatch"] = mism
        checks["P2_ref_group_c4map_vs_pool_mismatch"] = int((p2.ref_group != p2.reference.map(grp)).sum())
        checks["P2_query_group_c4map_vs_pool_mismatch"] = int((p2.query_group != p2["query"].map(grp)).sum())
        checks["P2_query_in_202"] = bool(p2["query"].isin(ann_ids).all())
        p2["roi_kind"] = "c4_labelled"
        # ---- P1
        qa = pd.read_csv(bound_path(bind, "query_assignment"), dtype=str).sort_values(
            "query_index", key=lambda c: c.astype(int))
        population = pd.DataFrame({"slide_id": qa.slide_id, "patient": qa.patient_id, "group": qa.diagnostic_group})
        pop_ids = list(population.slide_id)
        matrix = pd.read_pickle(bound_path(bind, "ot_matrix_all_tissue"))
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        sub = matrix.loc[pop_ids, pop_ids]
        topk = population_topk(sub.values, pop_ids, population.patient.values, 10)
        ref_info = pc[pc.loose][["slide_id", "patient", "group"]]
        mism = [r for r in loose if r in set(pop_ids) and
                population.set_index("slide_id").at[r, "patient"] != pat[r]]
        checks["P1_reference_patient_pool_vs_population_mismatch"] = mism
        p1 = pairs_p1(loose, ref_info, population, topk)
        # P1 ROIs
        qids = numeric_sorted(p1["query"].unique())
        lab_rows = {q: labelled_tumour_rows(q) for q in qids}
        pred_rows, tf2chk = tf2_predicted_rows(bind, qids)
        tf2chk.to_csv(d / "tf2_mask_checks.csv", index=False)
        rois = draw_rois(qids, lab_rows, pred_rows)
        p1["roi_labelled"] = p1["query"].map(lambda q: ";".join(map(str, rois[q].get("labelled", []))))
        p1["roi_predicted"] = p1["query"].map(lambda q: ";".join(map(str, rois[q].get("predicted", []))))
        p1["query_annotated"] = p1.roi_labelled != ""
        p1["roi_primary"] = np.where(p1.query_annotated, "labelled", np.where(p1.roi_predicted != "", "predicted", "none"))
        # ---- covariates & flags
        p1set = set(zip(p1.reference, p1["query"]))
        p2["also_P1"] = [(r, q) in p1set for r, q in zip(p2.reference, p2["query"])]
        pos = {s: i for i, s in enumerate(pop_ids)}

        ref_rank = {}
        for r in numeric_sorted(set(p2.reference)):
            if r not in pos:
                continue
            i = pos[r]
            sc = sub.values[i].astype(float).copy()
            patients = population.patient.values
            bad = (patients == patients[i]) | (sc >= 1e5)
            bad[i] = True
            sc[bad] = np.inf
            order = np.argsort(sc, kind="stable")
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1)
            ref_rank[r] = {pop_ids[j]: (int(ranks[j]) if np.isfinite(sc[j]) else None) for j in range(len(pop_ids))}
        p2["query_rank_in_reference_list"] = [ref_rank.get(r, {}).get(q) for r, q in zip(p2.reference, p2["query"])]
        p2["query_top10_of_reference"] = [(v is not None and v == v and v <= 10) for v in p2.query_rank_in_reference_list]
        # ---- split
        sp1 = split_patients(pc[pc.strict][["patient", "group"]])
        rest = pc[pc.loose & ~pc.patient.isin(sp1.patient)][["patient", "group"]]
        sp2 = split_patients(rest) if len(rest) else pd.DataFrame(columns=["patient", "group", "half"])
        sp1["set"] = "strict"
        sp2["set"] = "sensitivity_only"
        split = pd.concat([sp1, sp2], ignore_index=True)
        split.to_csv(d / "split.csv", index=False)
        half = dict(zip(split.patient, split.half))
        refs = pc[pc.loose][["slide_id", "patient", "group", "strict", "n_tumour_patches", "n_nontumour_patches"]].copy()
        refs["half"] = refs.patient.map(half)
        refs.to_csv(d / "references.csv", index=False)
        for df in (p1, p2):
            df["half"] = df.ref_patient.map(half)
            df["ref_strict"] = df.reference.isin(strict)
        # ---- pair ids, table
        p1 = p1.reset_index(drop=True)
        p2 = p2.reset_index(drop=True)
        p1["pair_id"] = [f"P1:{r}:{q}" for r, q in zip(p1.reference, p1["query"])]
        p2["pair_id"] = [f"P2:{r}:{arm}:{i}" for i, (r, arm) in enumerate(zip(p2.reference, p2.arm))]
        cols = ["pair_id", "design", "reference", "ref_patient", "ref_group", "ref_strict", "half", "arm", "query",
                "query_patient", "query_group"]
        p1[cols + ["rank_of_reference_in_query_list", "query_annotated", "roi_primary", "roi_labelled",
                   "roi_predicted"]].to_csv(d / "pairs_P1.csv", index=False)
        p2[cols + ["p2_source", "query_tiles", "also_P1", "query_rank_in_reference_list",
                   "query_top10_of_reference"]].to_csv(d / "pairs_P2.csv", index=False)
        # ---- F4 / F5 / F6
        ex20 = set(l.strip() for l in open(bound_path(bind, "excluded_20x")) if l.strip())
        pop_pat = dict(zip(population.slide_id, population.patient))
        f4_same = int((p1.ref_patient == p1.query_patient).sum() + (p2.ref_patient == p2.query_patient).sum())
        f4_same_pool = int(sum(pat.get(q, pop_pat.get(q)) == pat[r] for r, q in zip(p1.reference, p1["query"])))
        slides_used = set(p1.reference) | set(p1["query"]) | set(p2.reference) | set(p2["query"])
        f4_20x = numeric_sorted(slides_used & ex20)
        checks["F4"] = {"pass": f4_same == 0 and f4_same_pool == 0 and not f4_20x, "same_patient_pairs": f4_same,
                        "same_patient_pairs_cross_map": f4_same_pool, "slides_20x": f4_20x}
        dev = set(split[split.half == "development"].patient)
        rep = set(split[split.half == "report"].patient)
        checks["F5"] = {"pass": not (dev & rep) and split.patient.is_unique and refs.half.notna().all(),
                        "n_dev_patients": len(dev), "n_report_patients": len(rep),
                        "dev_strict_refs": int(((refs.half == "development") & refs.strict).sum()),
                        "report_strict_refs": int(((refs.half == "report") & refs.strict).sum())}
        rng = np.random.Generator(np.random.PCG64(SEED))
        f6q = [pop_ids[i] for i in sorted(rng.choice(len(pop_ids), size=20, replace=False))]
        f6 = []
        for q in f6q:
            ind = independent_topk(sub, q, pop_pat, 10)
            mine = [n for n, _ in topk[q]]
            p1_refs = set(p1[p1["query"] == q].reference)
            exp = {n for n in ind if n in set(loose) and pop_pat[q] != pat.get(n)}
            f6.append({"query": q, "topk_equal": ind == mine, "p1_refs_equal": p1_refs == exp,
                       "n_refs": len(exp)})
        checks["F6"] = {"pass": all(r["topk_equal"] and r["p1_refs_equal"] for r in f6), "queries": f6}
        # ---- counts (written before scoring)
        cnt1 = p1.groupby(["reference", "ref_group", "ref_strict", "half", "arm"]).size().unstack("arm", fill_value=0)
        cnt1.to_csv(d / "counts_P1_per_reference.csv")
        p1.groupby(["ref_group", "arm"]).size().unstack("arm", fill_value=0).to_csv(d / "counts_P1_per_group.csv")
        p2.groupby(["reference", "ref_group", "ref_strict", "half", "arm"]).size().unstack(
            "arm", fill_value=0).to_csv(d / "counts_P2_per_reference.csv")
        checks["counts"] = {"strict_refs": len(strict), "loose_refs": len(loose),
                            "strict_refs_in_population": int(sum(r in pos for r in strict)),
                            "P1_pairs": int(len(p1)), "P1_pairs_strict": int(p1.ref_strict.sum()),
                            "P1_same": int((p1.arm == "same_group").sum()),
                            "P1_other": int((p1.arm == "other_group").sum()),
                            "P1_queries": len(qids), "P1_queries_annotated": int(sum(bool(v) for v in lab_rows.values())),
                            "P1_roi_none": int((p1.roi_primary == "none").sum()),
                            "P2_pairs": int(len(p2)), "P2_pairs_strict": int(p2.ref_strict.sum()),
                            "P2_from_c4": int((p2.p2_source == "c4_draw").sum()),
                            "P2_continuation": int((p2.p2_source == "continuation").sum()),
                            "P2_also_P1": int(p2.also_P1.sum()),
                            "P2_also_P1_strict": int((p2.also_P1 & p2.ref_strict).sum()),
                            "unique_OT_plans": int(len(set(zip(p1.reference, p1["query"])) |
                                                       set(zip(p2.reference, p2["query"]))))}
        checks["strict_refs_vs_c4_44"] = {
            "c4_44_in_strict": int(len(set(pd.read_csv(bound_path(bind, "c4_whole_patient_per_reference"),
                                                       dtype=str).reference) & set(strict))),
            "strict_not_in_c4_44": numeric_sorted(set(strict) - set(pd.read_csv(
                bound_path(bind, "c4_whole_patient_per_reference"), dtype=str).reference))}
        (d / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
        ok = checks["F4"]["pass"] and checks["F5"]["pass"] and checks["F6"]["pass"]
        if not ok:
            fail(d, "s0_pairs", "F4/F5/F6 failure", checks)
            return 3
        finish(d, "s0_pairs", {"s0_bind": str(bind), "checks": checks})
        print(json.dumps(checks["counts"], indent=1), flush=True)
        return 0
    except Exception as e:  # pragma: no cover
        fail(d, "s0_pairs", repr(e), {"traceback": traceback.format_exc()})
        return 2


def stage_f1(a) -> int:
    d = new_run_dir("f1")
    try:
        bind = latest_complete("s0_bind")
        verify_complete(bind)
        LC, _ = import_prosper()
        meta = meta_cache_load(bind / "meta_tiles.pkl")
        # round_trip: pandas' default C float parser can be 1 ulp off the written repr (F1 run_001 lesson)
        c4 = pd.read_csv(C4_CSV, dtype={"reference": str, "query_slide": str, "query_tiles": str},
                         float_precision="round_trip")
        refs44 = pd.read_csv(C4_REFS, dtype={"reference": str}).reference.tolist()
        if len(refs44) != 44:
            raise RuntimeError(f"expected 44 C4 variant-A references, got {len(refs44)}")
        rows = c4[c4.reference.isin(refs44)].reset_index(drop=True)
        slides = {}

        def get(s):
            if s not in slides:
                slides[s] = LC.Slide(s, FEATURES, meta[s], LABELS)
            return slides[s]
        out, fails = [], []
        cols = ["auroc_A", "ap_A", "pointing_A", "tumour_fraction_A", "n_annotated",
                "auroc_B", "ap_B", "pointing_B", "tumour_fraction_B", "n_patches"]
        for k, r in rows.iterrows():
            ref, qs = get(r.reference), get(r.query_slide)
            qidx = np.array([int(x) for x in str(r.query_tiles).split(";")])
            qvec = qs.X[qidx].mean(axis=0)
            h = roi_map(ref, qvec, "l1", "mean")
            h_ref = ref.heatmap(qvec)
            same_map = bool(np.array_equal(h, h_ref))
            mine = score_continuous(h, ref.patch_label, LC)
            imported = LC.evaluate_pair(ref, qvec, C4_MIN_NEG)
            rec = {"reference": r.reference, "arm": r.arm, "query_slide": r.query_slide, "query_tiles": r.query_tiles,
                   "map_equals_imported_heatmap": same_map}
            ok = same_map
            for c in cols:
                c4v = r[c]
                v = mine.get(c, imported.get(c))
                if c in ("pointing_A", "pointing_B"):
                    c4v = None if pd.isna(c4v) else (str(c4v) == "True")
                    eq = (c4v is None and v is None) or (c4v == v)
                elif pd.isna(c4v):
                    eq = v is None or (isinstance(v, float) and np.isnan(v))
                else:
                    eq = v is not None and float(v) == float(c4v)
                rec[f"{c}_c4"] = c4v
                rec[f"{c}_rerun"] = v
                rec[f"{c}_equal"] = bool(eq)
                if c.endswith("_A") or c == "n_annotated":
                    ok &= bool(eq)
                if c in mine and c in imported:
                    ok &= (mine[c] == imported[c]) or (isinstance(mine[c], float) and np.isnan(mine[c]) and np.isnan(imported[c]))
            rec["pass"] = bool(ok)
            out.append(rec)
            if not ok:
                fails.append(k)
        df = pd.DataFrame(out)
        df.to_csv(d / "f1_per_pair.csv", index=False)
        summ = {"n_pairs": int(len(df)), "n_references": int(df.reference.nunique()),
                "n_same_group_pairs": int((df.arm == "same_group").sum()),
                "n_other_group_pairs": int((df.arm == "other_group").sum()),
                "n_fail": len(fails), "all_variant_A_equal": not fails,
                "variant_B_all_equal": bool(df[[c for c in df.columns if c.endswith("_B_equal")]].all().all()),
                "map_equals_imported_heatmap_all": bool(df.map_equals_imported_heatmap.all()),
                "comparison": "exact float equality vs the C4 per-pair CSV (float_precision=round_trip parse)",
                "c4_csv_sha256": sha256_file(C4_CSV)}
        (d / "f1_summary.json").write_text(json.dumps(summ, indent=1))
        if fails:
            fail(d, "f1", f"F1: {len(fails)} pair(s) differ from C4-ROI8", summ)
            return 3
        finish(d, "f1", {"pass": True, **summ, "s0_bind": str(bind)})
        print(f"[f1] PASS {summ}", flush=True)
        return 0
    except Exception as e:  # pragma: no cover
        fail(d, "f1", repr(e), {"traceback": traceback.format_exc()})
        return 2


def stage_f2(a) -> int:
    d = new_run_dir("f2")
    try:
        bind = latest_complete("s0_bind")
        verify_complete(bind)
        c4 = pd.read_csv(C4_CSV, dtype={"reference": str, "query_slide": str})
        refs44 = pd.read_csv(C4_REFS, dtype={"reference": str}).reference.tolist()
        cand = c4[c4.reference.isin(refs44)].drop_duplicates("reference").reset_index(drop=True)
        rng = np.random.Generator(np.random.PCG64(SEED))
        pick = cand.iloc[sorted(rng.choice(len(cand), size=5, replace=False))]
        matrix = pd.read_pickle(OT_MATRIX)
        gen = ot_generator()
        rows = []
        for _, r in pick.iterrows():
            Fq, _ = load_features(r.query_slide)
            Fr, _ = load_features(r.reference)
            t0 = time.time()
            T, M, dist = ot_plan(gen, Fq, Fr)
            el = time.time() - t0
            acc_rq, acc_qr = float(matrix.loc[r.reference, r.query_slide]), float(matrix.loc[r.query_slide, r.reference])
            rows.append({"reference": r.reference, "query": r.query_slide, "n_query": len(Fq), "n_ref": len(Fr),
                         "distance": dist, "sum_TM": float((T * M).sum()), "accepted_ref_row": acc_rq,
                         "accepted_query_row": acc_qr, "abs_diff_ref_row": abs(dist - acc_rq),
                         "abs_diff_query_row": abs(dist - acc_qr), "nnz": int((T > 0).sum()),
                         "row_marginal_maxerr": float(np.abs(T.sum(1) - 1 / len(Fq)).max()),
                         "col_marginal_maxerr": float(np.abs(T.sum(0) - 1 / len(Fr)).max()), "seconds": round(el, 2)})
        df = pd.DataFrame(rows)
        df.to_csv(d / "f2_pairs.csv", index=False)
        ok = bool((df.abs_diff_ref_row <= F2_TOL).all() and (df.abs_diff_query_row <= F2_TOL).all())
        summ = {"pass": ok, "tol": F2_TOL, "max_abs_diff": float(df[["abs_diff_ref_row", "abs_diff_query_row"]].max().max()),
                "pair_choice": "5 distinct C4 variant-A references (first C4 row each), PCG64(123) draw",
                "ot_code": "prosper OTSlideEmbeddingGenerator(metric='cosine', reg=0, feature_dim=1536).compute_ot_distance"}
        (d / "f2_summary.json").write_text(json.dumps(summ, indent=1))
        if not ok:
            fail(d, "f2", "F2: OT distance differs from the accepted matrix by > 1e-6", summ)
            return 3
        finish(d, "f2", {**summ, "s0_bind": str(bind)})
        print(f"[f2] PASS {summ}", flush=True)
        return 0
    except Exception as e:  # pragma: no cover
        fail(d, "f2", repr(e), {"traceback": traceback.format_exc()})
        return 2




# ---------------------------------------------------------------- F3 (Fig. 5 partitions; saved plans, no new plan)
def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal up to relabelling."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def notebook_partition(T: np.ndarray, res: float):
    """The Fig. 5 driver-notebook construction (string nodes A_i/B_j inserted in np.nonzero order)."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    G = nx.Graph()
    for i, j in zip(*np.nonzero(T)):
        G.add_edge(f"A_{i}", f"B_{j}", weight=float(T[i, j]))
    comms = louvain_communities(G, weight="weight", resolution=res, seed=0)
    la, lb = np.zeros(T.shape[0], dtype=int), np.zeros(T.shape[1], dtype=int)
    for cid, nodes in enumerate(comms, start=1):
        for u in nodes:
            side, k = u.split("_")
            (la if side == "A" else lb)[int(k)] = cid
    return la, lb


def stage_f3(a) -> int:
    d = new_run_dir("f3")
    try:
        from sklearn.metrics import adjusted_rand_score
        bind = latest_complete("s0_bind")
        verify_complete(bind)
        _, OV = import_prosper()
        out = []
        for panel, q, n, res in FIG5_PAIRS:
            dd = pickle.load(open(bound_path(bind, f"fig5_plan:{q}_{n}"), "rb"))
            T = np.asarray(dd["T"], dtype=np.float64)
            lab = pd.read_csv(bound_path(bind, f"fig5_labels:{q}_{n}"), dtype={"slide": str})
            pub_q = lab[lab.slide == q].sort_values("patch_idx").community.to_numpy()
            pub_n = lab[lab.slide == n].sort_values("patch_idx").community.to_numpy()
            pub = np.concatenate([pub_q, pub_n])
            rec = {"panel": panel, "query": q, "neighbour": n, "published_resolution": res,
                   "published_sparsification": "none (full plan)", "published_n_communities": int(len(set(pub)))}
            # (i) specification 001 statement: mutual top-5 + Louvain 1.0 via the imported function
            da, db = OV.cluster_patches_from_transport(T, method="louvain", resolution=1.0, sparsify_k=5)
            la = da.sort_values("patch_idx").normalized_score.to_numpy()
            lb = db.sort_values("patch_idx").normalized_score.to_numpy()
            x = np.concatenate([la, lb])
            rec.update({"mtop5_r1_imported_n_clusters": int(len(set(x))),
                        "mtop5_r1_imported_ARI_vs_published": float(adjusted_rand_score(pub, x)),
                        "mtop5_r1_imported_reproduces": same_partition(pub, x)})
            # (ii) this script's grid construction on the full plan at the published resolution
            cq, cr = cluster_sparse(T, "louvain", res)
            y = np.concatenate([cq, cr])
            rec.update({"full_plan_pubres_gridcode_n_clusters": int(len(set(y))),
                        "full_plan_pubres_gridcode_ARI": float(adjusted_rand_score(pub, y)),
                        "full_plan_pubres_gridcode_reproduces": same_partition(pub, y)})
            # (iii) the notebook construction at the published resolution: exact labels expected
            nq, nn = notebook_partition(T, res)
            rec.update({"notebook_construction_pubres_exact": bool(np.array_equal(nq, pub_q) and np.array_equal(nn, pub_n))})
            # (iv) grid resolutions: cluster counts only (does any grid resolution reproduce?)
            grid_hits = []
            for sp, k in (("none", None), ("mutual_topk", 5)):
                Tsp = sparsify(T, sp, k, OV)
                for r in LOUVAIN_RES:
                    gq, gr = cluster_sparse(Tsp, "louvain", r)
                    z = np.concatenate([gq, gr])
                    rec[f"grid_{sp}_{k}_r{r}_n_clusters"] = int(len(set(z)))
                    if same_partition(pub, z):
                        grid_hits.append(f"{sp}|{k}|{r}")
            rec["grid_settings_reproducing_published"] = ";".join(grid_hits)
            out.append(rec)
        df = pd.DataFrame(out)
        df.to_csv(d / "f3_fig5.csv", index=False)
        ok = bool(df.notebook_construction_pubres_exact.all())
        summ = {"pass": ok, "pass_rule": "the published partitions are reproduced exactly from the saved plans at the "
                "published recipe (full plan; Louvain 4e-5 RMS / 5e-4 RB; notebook graph construction)",
                "contract_statement_mtop5_r1_reproduces": bool(df.mtop5_r1_imported_reproduces.any()),
                "grid_code_full_plan_pubres_reproduces": df.full_plan_pubres_gridcode_reproduces.tolist(),
                "any_grid_setting_reproduces": bool((df.grid_settings_reproducing_published != "").any())}
        (d / "f3_summary.json").write_text(json.dumps(summ, indent=1))
        if not ok:
            fail(d, "f3", "F3: published Fig. 5 partitions not reproduced at the published recipe", summ)
            return 3
        finish(d, "f3", {**summ, "s0_bind": str(bind)})
        print(f"[f3] {summ}", flush=True)
        return 0
    except Exception as e:  # pragma: no cover
        fail(d, "f3", repr(e), {"traceback": traceback.format_exc()})
        return 2


# ---------------------------------------------------------------- S1 scoring kernels
def build_graph(T_sp: np.ndarray):
    import networkx as nx
    n_a, n_b = T_sp.shape
    G = nx.Graph()
    G.add_nodes_from(range(n_a), bipartite=0, slide='A')
    G.add_nodes_from(range(n_a, n_a + n_b), bipartite=1, slide='B')
    i_idx, j_idx = np.nonzero(T_sp)
    for i, j in zip(i_idx, j_idx):
        G.add_edge(i, n_a + j, weight=float(T_sp[i, j]))
    return G


def cluster_graph(G, n_a: int, n_b: int, method: str, resolution):
    import networkx as nx
    partition = {}
    if method == "connected_components":
        for cid, comp in enumerate(nx.connected_components(G), start=1):
            for u in comp:
                partition[u] = cid
    elif method == "louvain":
        from networkx.algorithms.community import louvain_communities
        for cid, nodes in enumerate(louvain_communities(G, weight="weight", resolution=resolution, seed=0), start=1):
            for u in nodes:
                partition[u] = cid
    else:
        raise ValueError(method)
    lab = np.array([partition[u] for u in range(n_a + n_b)], dtype=np.int64)
    return lab[:n_a], lab[n_a:]


def b_tile_values(T_sp: np.ndarray, M: np.ndarray, stat: str):
    """Reference-tile matched cost (lower = better) and linked mask; unlinked tiles -> max linked + 1 (choice 2)."""
    mass = T_sp.sum(axis=0)
    linked = mass > 0
    if stat == "meancost":
        val = np.full(T_sp.shape[1], np.nan)
        val[linked] = (T_sp * M).sum(axis=0)[linked] / mass[linked]
    elif stat == "mincost":
        val = np.where(T_sp > 0, M, np.inf).min(axis=0).astype(float)
        val[~linked] = np.nan
    else:
        raise ValueError(stat)
    if linked.any():
        val[~linked] = np.nanmax(val[linked]) + 1.0
    else:
        val[:] = 0.0
    return val, linked


def score_b(ref, T: np.ndarray, M: np.ndarray, LC, OV) -> list:
    out = []
    T_cache = {}
    for st in grid_b():
        key = (st["sparsify"], st["k"])
        if key not in T_cache:
            T_cache[key] = sparsify(T, st["sparsify"], st["k"], OV)
        T_sp = T_cache[key]
        val, linked = b_tile_values(T_sp, M, st["stat"])
        tv = val[ref.rows]
        h = normalise_invert(project(tv, ref.owners, "mean"))
        rec = score_continuous(h, ref.patch_label, LC, min_neg=1)
        ann = (ref.patch_label == TUMOR) | np.isin(ref.patch_label, NONTUMOR)
        lk = linked[ref.rows]
        cov_patch = np.array([lk[o].any() for o in ref.owners])
        rec.update({"family": "B", "setting": st["setting"], "coverage_tiles": float(linked.mean()),
                    "coverage_annotated_patches": float(cov_patch[ann].mean()) if ann.any() else float("nan")})
        out.append(rec)
    return out


class LouvainTimeout(Exception):
    pass


def _alarm(*_):
    raise LouvainTimeout()


def score_c(T: np.ndarray, lab_q: np.ndarray, lab_r: np.ndarray, OV, timeout_s: int = LOUVAIN_TIMEOUT_S) -> list:
    """Family C grid on one plan.  A clustering that does not return within timeout_s wall seconds (networkx
    louvain_communities can oscillate forever on these graphs; see [internal record]) is recorded with
    terminated=False and NaN cluster metrics (addendum choice A1)."""
    import signal
    out = []
    n_a, n_b = T.shape
    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        for sp, p in c_sparsifications():
            T_sp = sparsify(T, sp, p, OV)
            agb = cross_slide_agreement(T_sp, lab_q, lab_r, binary=True)
            ag5 = cross_slide_agreement(T_sp, lab_q, lab_r, binary=False)
            G = build_graph(T_sp)
            for cl, r in c_clusterings():
                t0 = time.time()
                signal.alarm(int(timeout_s))
                try:
                    cq, cr = cluster_graph(G, n_a, n_b, cl, r)
                    ok = True
                except LouvainTimeout:
                    ok = False
                finally:
                    signal.alarm(0)
                secs = round(time.time() - t0, 3)
                rec = {"family": "C", "setting": f"C|{sp}|{p}|{cl}|{r}",
                       "agreement_bin": agb["agreement"], "chance_bin": agb["chance"],
                       "excess_bin": agb["agreement"] - agb["chance"], "annotated_mass_bin": agb["annotated_mass"],
                       "agreement_5": ag5["agreement"], "chance_5": ag5["chance"],
                       "excess_5": ag5["agreement"] - ag5["chance"], "edges": int(np.count_nonzero(T_sp)),
                       "terminated": ok, "cluster_seconds": secs}
                if ok:
                    sb = cluster_stats(T_sp, cq, cr, lab_q, lab_r, binary=True)
                    s5 = cluster_stats(T_sp, cq, cr, lab_q, lab_r, binary=False)
                    rec.update({"purity_bin": sb["purity"], "purity_5": s5["purity"], "nmi_bin": sb["nmi_ref"],
                                "nmi_5": s5["nmi_ref"], "n_clusters": sb["n_clusters"],
                                "median_cluster_size": sb["median_cluster_size"],
                                "frac_singleton_clusters": sb["frac_singleton_clusters"],
                                "degenerate_pair": bool(sb["n_clusters"] == 1 or sb["frac_singleton_clusters"] > 0.5)})
                else:
                    rec.update({k: float("nan") for k in ("purity_bin", "purity_5", "nmi_bin", "nmi_5", "n_clusters",
                                                           "median_cluster_size", "frac_singleton_clusters")})
                    rec["degenerate_pair"] = True
                out.append(rec)
    finally:
        signal.signal(signal.SIGALRM, old)
    return out


def score_a(ref_slide, qvec: np.ndarray, LC) -> list:
    out = []
    for st in grid_a():
        h = roi_map(ref_slide, qvec, st["distance"], st["aggregation"])
        rec = score_continuous(h, ref_slide.patch_label, LC, min_neg=1)
        rec.update({"family": "A", "setting": st["setting"]})
        out.append(rec)
    return out


def parse_idx(s) -> np.ndarray:
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s) == "":
        return np.zeros(0, dtype=np.int64)
    return np.array([int(x) for x in str(s).split(";")], dtype=np.int64)


def stage_s1_cmds(a) -> int:
    s0 = latest_complete("s0_pairs")
    verify_complete(s0)
    p1 = pd.read_csv(s0 / "pairs_P1.csv", dtype=str)
    p2 = pd.read_csv(s0 / "pairs_P2.csv", dtype=str)
    refs = numeric_sorted(set(p1.reference) | set(p2.reference))
    d = Path(a.run_dir) if a.run_dir else new_run_dir("s1")
    py = "/HOME/miniconda3/bin/python3"
    lines = [f"export PROSPER_OUT={OUT} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; "
             f"{py} {Path(__file__).resolve()} s1 --reference {r} --s0 {s0} --run-dir {d}"
             + (" --skip-c" if a.skip_c else "") for r in refs]
    if a.only:
        lines = [l for l, r in zip(lines, refs) if r in set(a.only.split(","))]
    name = a.cmds_name or "cmds.txt"
    if (d / name).exists():
        raise RuntimeError(f"{d / name} exists (never overwritten)")
    (d / name).write_text("\n".join(lines) + "\n")
    (d / (Path(name).stem + "_PLAN.json")).write_text(json.dumps({"created": now(), "s0_pairs": str(s0), "n_tasks": len(lines),
                                                "script_sha256": sha256_file(Path(__file__).resolve())}, indent=1))
    print(d)
    return 0


def stage_s1(a) -> int:
    s0 = Path(a.s0)
    run = Path(a.run_dir)
    r = a.reference
    od = run / "refs" / r
    if (od / "DONE.json").exists():
        print(f"[s1] {r} already done")
        return 0
    if od.exists():                                  # preserve a failed/partial attempt
        k = 1
        while (run / "refs" / f"{r}.attempt{k:02d}").exists():
            k += 1
        od.rename(run / "refs" / f"{r}.attempt{k:02d}")
    od.mkdir(parents=True)
    (od / "plans").mkdir()
    t_start = time.time()
    try:
        verify_complete(s0)
        bind = Path(json.loads((s0 / "COMPLETE.json").read_text())["s0_bind"])
        LC, OV = import_prosper()
        meta = meta_cache_load(bind / "meta_tiles.pkl")
        p1 = pd.read_csv(s0 / "pairs_P1.csv", dtype=str)
        p2 = pd.read_csv(s0 / "pairs_P2.csv", dtype=str)
        p1, p2 = p1[p1.reference == r], p2[p2.reference == r]
        matrix = pd.read_pickle(bound_path(bind, "ot_matrix_all_tissue"))
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        ref = MapSlide(r, meta[r], LC=LC)
        gen = ot_generator()
        p2q = set(p2["query"])
        queries = numeric_sorted(set(p1["query"]) | p2q)
        plans, bc_rows, a_rows = [], [], []
        for q in queries:
            t0 = time.time()
            Fq, qc = load_features(q)
            T, M, dist = ot_plan(gen, Fq, ref.F)
            ii, jj = np.nonzero(T)
            np.savez_compressed(od / "plans" / f"{q}.npz", i=ii.astype(np.int32), j=jj.astype(np.int32),
                                t=T[ii, jj], m=np.asarray(M)[ii, jj].astype(np.float32),
                                shape=np.array(T.shape), distance=np.array([dist]))
            acc = float(matrix.loc[r, q]) if (r in matrix.index and q in matrix.columns) else float("nan")
            for rec in score_b(ref, T, M, LC, OV):
                bc_rows.append({"reference": r, "query": q, **rec})
            if q in p2q and not getattr(a, "skip_c", False):
                lab = load_labels(q)
                lab_q = np.array([lab.get(c, 0) for c in qc], dtype=np.int8)
                for rec in score_c(T, lab_q, ref.row_label, OV):
                    bc_rows.append({"reference": r, "query": q, **rec})
            plans.append({"reference": r, "query": q, "n_query": int(T.shape[0]), "n_ref": int(T.shape[1]),
                          "distance": dist, "matrix_ref_row": acc, "abs_diff": abs(dist - acc), "nnz": int(len(ii)),
                          "seconds": round(time.time() - t0, 2)})
            del T, M
            # family A for every pair row with this query (P1: feature-row ROIs; P2: C4 Slide.X ROIs)
            for _, row in p1[p1["query"] == q].iterrows():
                for kind in ("labelled", "predicted"):
                    idx = parse_idx(row[f"roi_{kind}"])
                    if len(idx) == 0:
                        continue
                    for rec in score_a(ref.slide, Fq[idx].mean(axis=0), LC):
                        a_rows.append({"pair_id": row.pair_id, "roi_kind": kind,
                                       "roi_primary": kind == row.roi_primary, **rec})
            if q in p2q:
                qs = LC.Slide(q, FEATURES, meta[q], LABELS)
                for _, row in p2[p2["query"] == q].iterrows():
                    for rec in score_a(ref.slide, qs.X[parse_idx(row.query_tiles)].mean(axis=0), LC):
                        a_rows.append({"pair_id": row.pair_id, "roi_kind": "c4_labelled", "roi_primary": True, **rec})
                del qs
        pd.DataFrame(a_rows).to_csv(od / "a_scores.csv.gz", index=False)
        pd.DataFrame(bc_rows).to_csv(od / "bc_scores.csv.gz", index=False)
        pd.DataFrame(plans).to_csv(od / "plans.csv", index=False)
        outputs = {p.relative_to(od).as_posix(): sha256_file(p) for p in sorted(od.rglob("*")) if p.is_file()}
        (od / "DONE.json").write_text(json.dumps({"reference": r, "finished": now(), "host": platform.node(),
                                                  "seconds": round(time.time() - t_start, 1), "n_queries": len(queries),
                                                  "n_p1": int(len(p1)), "n_p2": int(len(p2)),
                                                  "script_sha256": sha256_file(Path(__file__).resolve()),
                                                  "outputs": outputs}, indent=1))
        print(f"[s1] {r}: {len(queries)} plans in {time.time() - t_start:.0f}s", flush=True)
        return 0
    except Exception as e:
        (od / "FAILED.json").write_text(json.dumps({"reference": r, "failed": now(), "reason": repr(e),
                                                    "traceback": traceback.format_exc()}, indent=1))
        print(traceback.format_exc(), flush=True)
        return 2


# ---------------------------------------------------------------- C stage (amendment 003): family C from saved plans
ADDENDUM_001 = """# R3.5-L1 implementation choices -- addendum 001 (implementer, written before any family-C score of the
# amended grid exists; bound by sha256 in the C stage's COMPLETE.json)

A1. Non-termination. networkx 3.5 louvain_communities (the imported clustering) does not terminate on some
    (pair, setting) graphs: its first-level move loop oscillates on exactly tied modularity gains (observed in
    S1 run_001: 10 tasks stalled > 70 min; diagnostic DIAG_louvain_nontermination/, 1-17 of 190 settings per
    stalled pair failed a 30 s limit; normal calls take < 1 s). Rule: each clustering call gets 30 s wall; a
    call that does not return is recorded terminated=False, its cluster metrics are NaN, the pair counts as
    degenerate for that setting, and the pair is excluded from every family-C mean for that setting
    (agreement - chance included). The algorithm, seed and weights are not altered.
A2. Amendment 003 grid: Louvain resolutions {1e-5, 4e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1} + the
    contract-001 set (19 resolutions; 4e-5 is included because it is the submitted reference value and the
    amendment's count of 19 requires it) x 19 sparsifications + connected components = 380 settings. Grid order
    for tie-breaks = sparsification order, then connected components, then ascending resolution.
A3. Submitted C setting for the outcome rule on P2 pairs = C|none|None|louvain|4e-05 (amendment 003).
    In S3 under the submitted setting the Fig. 5 pairs use their published partitions (notebook construction,
    reproduced exactly by F3: RMS 4e-5, RB 5e-4); annotated pairs use C|none|None|louvain|4e-05 with the grid
    construction.
A4. Family C is scored from the S1 saved plans (nonzero entries stored in float64; the dense plan is rebuilt
    exactly). The 36 references whose S1 run also scored the 190-setting grid are re-scored here; S2 checks
    that the re-scored values equal the S1 values on every overlapping setting.
"""


NPY_PIN = "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"

ADDENDUM_002 = """# R3.5-L1 implementation choices -- addendum 002 (implementer; written before the re-scored B/C values exist)

B1. Host-dependent tie-breaking. ot_visualization.sparsify_transport_matrix selects top-k entries with
    np.argpartition; exact plans contain many exactly tied masses (1/n), and numpy 2.2 dispatches argpartition to
    an AVX-512 kernel on AVX-512 hosts that breaks ties differently from the AVX2 kernel (diag_argpartition_ties/:
    17 of 24 top-k masks differ on nodelmr13 vs this AVX2 host; 0 differ on nodelmr16 with
    NPY_DISABLE_CPU_FEATURES set to the AVX-512 features). S1 B scores and C-stage run_001 C scores therefore
    depend on which host ran them (found by the S2 A4 check, s2/run_001 FAILED).
B2. Remedy (environment only; code, settings and algorithm unchanged): families B and C are re-scored from the
    saved S1 plans with NPY_DISABLE_CPU_FEATURES set to the AVX-512 feature list on every host, which makes
    the imported top-k use the AVX2 kernel everywhere; the run refuses to start if AVX512_SKX is still active.
    Family A uses no partition/quicksort and F1 reproduced C4 exactly; A scores from S1 are kept.
B3. The saved plans store the nonzero masses (float64) and their cosine costs (float32, exactly the values of
    ot.dist); B uses costs only where the sparsified plan is nonzero, so the dense rebuild is exact.
"""


def assert_pinned_dispatch():
    from numpy._core._multiarray_umath import __cpu_features__ as cf
    if any(cf.get(k) for k in ("AVX512_SKX", "AVX512F")):
        raise RuntimeError("AVX-512 dispatch active: set NPY_DISABLE_CPU_FEATURES (addendum 002 B2)")
    return {"NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
            "avx2": bool(cf.get("AVX2"))}


def load_plan_with_costs(path: Path):
    z = np.load(path)
    shape = tuple(int(v) for v in z["shape"])
    T = np.zeros(shape, dtype=np.float64)
    T[z["i"], z["j"]] = z["t"]
    Mc = np.zeros(shape, dtype=np.float32)
    Mc[z["i"], z["j"]] = z["m"]
    return T, Mc


def load_plan(path: Path) -> np.ndarray:
    z = np.load(path)
    T = np.zeros(tuple(int(v) for v in z["shape"]), dtype=np.float64)
    T[z["i"], z["j"]] = z["t"]
    return T


def stage_c_cmds(a) -> int:
    s0 = latest_complete("s0_pairs")
    verify_complete(s0)
    s1 = Path(a.s1) if a.s1 else sorted((ROOT / "s1").glob("run_*"))[-1]
    p2 = pd.read_csv(s0 / "pairs_P2.csv", dtype=str)
    p1 = pd.read_csv(s0 / "pairs_P1.csv", dtype=str)
    refs = numeric_sorted(set(p2.reference) | set(p1.reference))
    missing = [r for r in refs if not (s1 / "refs" / r / "DONE.json").exists()]
    if missing:
        raise RuntimeError(f"S1 incomplete for {missing}")
    d = new_run_dir("c_stage")
    (d / "IMPLEMENTATION_CHOICES_ADDENDUM_001.md").write_text(ADDENDUM_001)
    (d / "IMPLEMENTATION_CHOICES_ADDENDUM_002.md").write_text(ADDENDUM_002)
    py = "/HOME/miniconda3/bin/python3"
    lines = [f"export PROSPER_OUT={OUT} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
             f"NPY_DISABLE_CPU_FEATURES='{NPY_PIN}'; "
             f"{py} {Path(__file__).resolve()} c --reference {r} --s0 {s0} --s1 {s1} --run-dir {d}" for r in refs]
    (d / "cmds.txt").write_text("\n".join(lines) + "\n")
    (d / "C_PLAN.json").write_text(json.dumps({"created": now(), "s0_pairs": str(s0), "s1": str(s1),
                                               "n_tasks": len(lines), "n_settings": len(grid_c()),
                                               "addendum_sha256": sha256_file(d / "IMPLEMENTATION_CHOICES_ADDENDUM_001.md"),
                                               "script_sha256": sha256_file(Path(__file__).resolve())}, indent=1))
    print(d)
    return 0


def stage_c(a) -> int:
    s0, s1, run, r = Path(a.s0), Path(a.s1), Path(a.run_dir), a.reference
    od = run / "refs" / r
    if (od / "DONE.json").exists():
        print(f"[c] {r} already done")
        return 0
    if od.exists():
        k = 1
        while (run / "refs" / f"{r}.attempt{k:02d}").exists():
            k += 1
        od.rename(run / "refs" / f"{r}.attempt{k:02d}")
    od.mkdir(parents=True)
    t_start = time.time()
    try:
        pin = assert_pinned_dispatch()
        verify_complete(s0)
        LC, OV = import_prosper()
        bind = Path(json.loads((s0 / "COMPLETE.json").read_text())["s0_bind"])
        rec1 = json.loads((s1 / "refs" / r / "DONE.json").read_text())
        p2 = pd.read_csv(s0 / "pairs_P2.csv", dtype=str)
        p2 = p2[p2.reference == r]
        p1 = pd.read_csv(s0 / "pairs_P1.csv", dtype=str)
        p1 = p1[p1.reference == r]
        meta = meta_cache_load(bind / "meta_tiles.pkl")
        ref = MapSlide(r, meta[r], LC=LC)
        rc = ref.coords
        lab_r = ref.row_label
        rows, brows = [], []
        p2q = set(p2["query"])
        for q in numeric_sorted(p2q | set(p1["query"])):
            pp = s1 / "refs" / r / "plans" / f"{q}.npz"
            if sha256_file(pp) != rec1["outputs"][f"plans/{q}.npz"]:
                raise RuntimeError(f"{pp}: sha256 differs from S1 DONE.json")
            T, Mc = load_plan_with_costs(pp)
            _, qc = load_features(q)
            if T.shape != (len(qc), len(rc)):
                raise RuntimeError(f"{pp}: plan shape {T.shape} != features {(len(qc), len(rc))}")
            for x in score_b(ref, T, Mc, LC, OV):
                brows.append({"reference": r, "query": q, **x})
            if q in p2q:
                labq = load_labels(q)
                lab_q = np.array([labq.get(c, 0) for c in qc], dtype=np.int8)
                for x in score_c(T, lab_q, lab_r, OV):
                    rows.append({"reference": r, "query": q, **x})
            del T, Mc
        df = pd.DataFrame(rows)
        df.to_csv(od / "c_scores.csv.gz", index=False)
        pd.DataFrame(brows).to_csv(od / "b_scores.csv.gz", index=False)
        outputs = {"c_scores.csv.gz": sha256_file(od / "c_scores.csv.gz"),
                   "b_scores.csv.gz": sha256_file(od / "b_scores.csv.gz")}
        (od / "DONE.json").write_text(json.dumps({"reference": r, "finished": now(), "host": platform.node(),
                                                  "seconds": round(time.time() - t_start, 1),
                                                  "n_queries": int(p2["query"].nunique()),
                                                  "n_nonterminated": int((~df.terminated.astype(bool)).sum()) if len(df) else 0,
                                                  "dispatch": pin,
                                                  "script_sha256": sha256_file(Path(__file__).resolve()),
                                                  "outputs": outputs}, indent=1))
        print(f"[c] {r}: {len(df)} rows in {time.time() - t_start:.0f}s", flush=True)
        return 0
    except Exception as e:
        (od / "FAILED.json").write_text(json.dumps({"reference": r, "failed": now(), "reason": repr(e),
                                                    "traceback": traceback.format_exc()}, indent=1))
        print(traceback.format_exc(), flush=True)
        return 2


def stage_c_finish(a) -> int:
    """Seal a C-stage run: every reference DONE and hash-consistent -> COMPLETE.json (relative outputs)."""
    run = Path(a.run_dir) if a.run_dir else sorted((ROOT / "c_stage").glob("run_*"))[-1]
    plan = json.loads((run / "C_PLAN.json").read_text())
    s0 = Path(plan["s0_pairs"])
    refs = numeric_sorted(set(pd.read_csv(s0 / "pairs_P2.csv", dtype=str).reference) |
                          set(pd.read_csv(s0 / "pairs_P1.csv", dtype=str).reference))
    outputs, secs, nonterm, hashes = {}, 0.0, 0, set()
    for r in refs:
        rec = json.loads((run / "refs" / r / "DONE.json").read_text())
        for name, h in rec["outputs"].items():
            if sha256_file(run / "refs" / r / name) != h:
                raise RuntimeError(f"{r}/{name}: sha256 differs")
            outputs[f"refs/{r}/{name}"] = h
        secs += rec["seconds"]
        nonterm += rec["n_nonterminated"]
        hashes.add(rec["script_sha256"])
    for f in ("IMPLEMENTATION_CHOICES_ADDENDUM_001.md", "IMPLEMENTATION_CHOICES_ADDENDUM_002.md", "cmds.txt",
              "C_PLAN.json"):
        if (run / f).exists():
            outputs[f] = sha256_file(run / f)
    rec = {"stage": "c_stage", "finished": now(), "script_sha256_finish": sha256_file(Path(__file__).resolve()),
           "task_script_sha256": sorted(hashes), "n_references": len(refs), "cpu_seconds": round(secs, 1),
           "n_nonterminated_rows": nonterm, "outputs": outputs, **plan}
    (run / "COMPLETE.json").write_text(json.dumps(rec, indent=1))
    print(f"[c-finish] {run}: {len(refs)} refs, {secs/3600:.2f} CPU-h, {nonterm} non-terminated rows")
    return 0


# ---------------------------------------------------------------- S2
A_METRICS = ["auroc_A", "ap_A", "pointing_A", "tumour_fraction_A", "pointing_minus_share"]
C_METRICS = ["excess_bin", "agreement_bin", "chance_bin", "purity_bin", "nmi_bin", "n_clusters",
             "median_cluster_size", "excess_5", "purity_5", "nmi_5"]
PRIMARY = {"A": "auroc_A", "B": "auroc_A", "C": "excess_bin"}
SELECTION_DESIGN = {"A": "P1", "B": "P1", "C": "P2"}
REPORT_DESIGNS = {"A": ("P1", "P2"), "B": ("P1", "P2"), "C": ("P2",)}


def replicate_means(values, patients, pats, mult) -> np.ndarray:
    v = np.asarray(values, float)
    W = mult[:, pd.Index(pats).get_indexer(list(patients))].astype(float)
    den = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (W @ v) / np.where(den > 0, den, 1), np.nan)


def summarise_reps(point: float, reps: np.ndarray, n_refs: int) -> dict:
    fin = reps[np.isfinite(reps)]
    return {"point": float(point), "ci95_lo": float(np.percentile(fin, 2.5)) if len(fin) else None,
            "ci95_hi": float(np.percentile(fin, 97.5)) if len(fin) else None, "n_references": int(n_refs),
            "n_replicates_finite": int(len(fin))}


def per_reference(df: pd.DataFrame, metrics) -> pd.DataFrame:
    keys = ["design", "family", "setting", "arm", "reference", "ref_patient", "ref_strict", "half", "ref_group"]
    return df.groupby(keys, dropna=False)[metrics].mean().reset_index()


def collect_s1(s1: Path, refs) -> tuple:
    a, bc, pl, done = [], [], [], {}
    for r in refs:
        od = s1 / "refs" / r
        if not (od / "DONE.json").exists():
            raise RuntimeError(f"S1 reference {r} has no DONE.json")
        rec = json.loads((od / "DONE.json").read_text())
        for name, h in rec["outputs"].items():
            if sha256_file(od / name) != h:
                raise RuntimeError(f"{od}/{name}: sha256 differs from DONE.json")
        done[r] = {"seconds": rec["seconds"], "host": rec["host"], "script_sha256": rec["script_sha256"]}
        a.append(pd.read_csv(od / "a_scores.csv.gz", dtype={"pair_id": str}, float_precision="round_trip"))
        bc.append(pd.read_csv(od / "bc_scores.csv.gz", dtype={"reference": str, "query": str},
                              float_precision="round_trip"))
        pl.append(pd.read_csv(od / "plans.csv", dtype={"reference": str, "query": str}, float_precision="round_trip"))
    return pd.concat(a, ignore_index=True), pd.concat(bc, ignore_index=True), pd.concat(pl, ignore_index=True), done


def stage_s2(a) -> int:
    d = new_run_dir("s2")
    try:
        s0 = latest_complete("s0_pairs")
        verify_complete(s0)
        s1 = Path(a.s1) if a.s1 else sorted((ROOT / "s1").glob("run_*"))[-1]
        f1 = latest_complete("f1")
        f2 = latest_complete("f2")
        f3 = latest_complete("f3")
        for x in (f1, f2, f3):
            verify_complete(x)
        bind = Path(json.loads((s0 / "COMPLETE.json").read_text())["s0_bind"])
        p1 = pd.read_csv(s0 / "pairs_P1.csv", dtype=str)
        p2 = pd.read_csv(s0 / "pairs_P2.csv", dtype=str)
        for p in (p1, p2):
            p["ref_strict"] = p.ref_strict == "True"
        refs = numeric_sorted(set(p1.reference) | set(p2.reference))
        A, BC, PL, done = collect_s1(s1, refs)
        checks = {}
        # F2b: every plan's distance vs the accepted matrix
        checks["F2b_all_plans"] = {"n_plans": int(len(PL)), "max_abs_diff": float(PL.abs_diff.max()),
                                   "pass": bool((PL.abs_diff <= F2_TOL).all())}
        # F1b: P2 = C4 pairs; submitted ROI map equals C4 per pair on the 44 C4 references
        c4 = pd.read_csv(bound_path(bind, "c4_roi8_per_pair"), dtype={"reference": str, "query_slide": str},
                         float_precision="round_trip")
        refs44 = set(pd.read_csv(bound_path(bind, "c4_whole_patient_per_reference"), dtype=str).reference)
        c4["c4_row"] = np.arange(len(c4))
        p2c = p2[p2.p2_source == "c4_draw"].copy()
        a_sub = A[(A.setting == SUBMITTED["A"]) & (A.roi_kind == "c4_labelled")].set_index("pair_id")
        # C4 rows in order per reference/arm -> P2 rows in the same order (both are the verbatim draw)
        c4_44 = c4[c4.reference.isin(refs44)].reset_index(drop=True)
        p2_44 = p2c[p2c.reference.isin(refs44)].reset_index(drop=True)
        same_rows = (len(c4_44) == len(p2_44) and (c4_44.reference.values == p2_44.reference.values).all()
                     and (c4_44.query_slide.values == p2_44["query"].values).all())
        f1b_ok = same_rows
        n_cmp = 0
        if same_rows:
            for k in range(len(c4_44)):
                rr = a_sub.loc[p2_44.pair_id[k]]
                n_cmp += 1
                f1b_ok &= (float(rr.auroc_A) == float(c4_44.auroc_A[k])
                           and str(rr.pointing_A) == str(c4_44.pointing_A[k]))
        checks["F1b_P2_submitted_equals_C4"] = {"pass": bool(f1b_ok), "n_pairs_compared": n_cmp,
                                                "n_c4_references_in_P2": int(p2_44.reference.nunique())}
        checks["F1"] = json.loads((f1 / "COMPLETE.json").read_text()).get("pass")
        checks["F2"] = json.loads((f2 / "COMPLETE.json").read_text()).get("pass")
        checks["F3"] = json.loads((f3 / "COMPLETE.json").read_text()).get("pass")
        if not (checks["F2b_all_plans"]["pass"] and f1b_ok and checks["F1"] and checks["F2"] and checks["F3"]):
            (d / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
            fail(d, "s2", "fidelity failure in S2 (F1b/F2b/F1-F3)", checks)
            return 3
        # ---- per-pair long tables
        pcols = ["pair_id", "design", "reference", "ref_patient", "ref_group", "ref_strict", "half", "arm", "query"]
        pairs = pd.concat([p1[pcols], p2[pcols]], ignore_index=True)
        A = A[A.roi_primary.astype(str) == "True"].merge(pairs, on="pair_id", how="inner")
        bc = BC.merge(pairs, on=["reference", "query"], how="inner")
        # families B and C: re-scored from the saved plans with pinned SIMD dispatch (addenda 001 A4, 002 B1-B2)
        crun = Path(a.c_run) if a.c_run else latest_complete("c_stage")
        verify_complete(crun)
        cprev = Path(a.c_prev) if a.c_prev else None
        cparts, bparts, chost = [], [], {}
        for r in refs:
            od = crun / "refs" / r
            rec = json.loads((od / "DONE.json").read_text())
            if not rec.get("dispatch") or rec["dispatch"].get("NPY_DISABLE_CPU_FEATURES") is None:
                raise RuntimeError(f"{od}: not run with pinned dispatch")
            for name in ("c_scores.csv.gz", "b_scores.csv.gz"):
                if sha256_file(od / name) != rec["outputs"][name]:
                    raise RuntimeError(f"{od}/{name}: sha256 differs")
            cparts.append(pd.read_csv(od / "c_scores.csv.gz", dtype={"reference": str, "query": str},
                                      float_precision="round_trip"))
            bparts.append(pd.read_csv(od / "b_scores.csv.gz", dtype={"reference": str, "query": str},
                                      float_precision="round_trip"))
        CN = pd.concat(cparts, ignore_index=True)
        CN["terminated"] = CN.terminated.astype(str) == "True"
        CN["degenerate_pair"] = CN.degenerate_pair.astype(str) == "True"
        BN = pd.concat(bparts, ignore_index=True)

        def compare(old, new, keys, cols, hosts):
            mm = old.merge(new, on=keys, suffixes=("_o", "_n"))
            per = {}
            for r, g in mm.groupby("reference"):
                bad = 0
                for c_ in cols:
                    x = g[f"{c_}_o"].astype(float).values
                    y = g[f"{c_}_n"].astype(float).values
                    bad += int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())
                per[r] = {"rows": int(len(g)), "mismatching_values": bad, "earlier_host": hosts.get(r)}
            avx2 = {r: v for r, v in per.items() if str(v["earlier_host"]).startswith(("noderome", "nodegpu247"))}
            return {"pass_on_avx2_hosts": all(v["mismatching_values"] == 0 for v in avx2.values()),
                    "n_refs_avx2_hosts": len(avx2), "n_refs_equal": sum(v["mismatching_values"] == 0 for v in per.values()),
                    "n_refs": len(per), "per_reference": per}
        s1host = {r: v["host"] for r, v in done.items()}
        bcols = ["auroc_A", "ap_A", "tumour_fraction_A", "coverage_tiles", "coverage_annotated_patches"]
        checks["B_rescore_vs_S1"] = compare(BC[BC.family == "B"], BN, ["reference", "query", "setting"], bcols, s1host)
        ccols = ["agreement_bin", "chance_bin", "excess_bin", "agreement_5", "excess_5", "purity_bin", "purity_5",
                 "nmi_bin", "nmi_5", "n_clusters", "median_cluster_size", "frac_singleton_clusters", "edges"]
        checks["C_rescore_vs_S1_190grid"] = compare(BC[BC.family == "C"], CN, ["reference", "query", "setting"], ccols, s1host)
        if cprev is not None:
            prev = pd.concat([pd.read_csv(cprev / "refs" / r / "c_scores.csv.gz", dtype={"reference": str, "query": str},
                                          float_precision="round_trip") for r in refs if (cprev / "refs" / r / "c_scores.csv.gz").exists()])
            phost = {r: json.loads((cprev / "refs" / r / "DONE.json").read_text())["host"] for r in refs
                     if (cprev / "refs" / r / "DONE.json").exists()}
            checks["C_rescore_vs_cstage_run001"] = compare(prev, CN, ["reference", "query", "setting"], ccols, phost)
        checks["C_nonterminated"] = {"rows": int((~CN.terminated).sum()), "of": int(len(CN)),
                                     "settings_affected": int(CN[~CN.terminated].setting.nunique()),
                                     "pairs_affected": int(CN[~CN.terminated][["reference", "query"]].drop_duplicates().shape[0])}
        det_ok = checks["B_rescore_vs_S1"]["pass_on_avx2_hosts"] and checks["C_rescore_vs_S1_190grid"]["pass_on_avx2_hosts"] \
            and (cprev is None or checks["C_rescore_vs_cstage_run001"]["pass_on_avx2_hosts"])
        if not det_ok:
            (d / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
            fail(d, "s2", "re-scored B/C differ from earlier AVX2-host values (determinism check)", checks)
            return 3
        for c_ in ("agreement_bin", "chance_bin", "excess_bin", "agreement_5", "chance_5", "excess_5"):
            CN.loc[~CN.terminated, c_] = np.nan                        # addendum A1
        Ct = CN.merge(pairs, on=["reference", "query"], how="inner")
        Ct = Ct[Ct.design == "P2"].copy()
        Bt = BN.merge(pairs, on=["reference", "query"], how="inner")
        for t in (A, Bt):
            t["pointing_A"] = t.pointing_A.map(lambda v: float(str(v) == "True") if pd.notna(v) else np.nan)
            t["pointing_minus_share"] = t.pointing_A - t.tumour_fraction_A
        A.to_csv(d / "per_pair_A.csv.gz", index=False)
        Bt.to_csv(d / "per_pair_B.csv.gz", index=False)
        Ct.to_csv(d / "per_pair_C.csv.gz", index=False)
        fam_tab = {"A": A, "B": Bt, "C": Ct}
        fam_metrics = {"A": A_METRICS, "B": A_METRICS + ["coverage_tiles", "coverage_annotated_patches"],
                       "C": C_METRICS}
        perref = {f: per_reference(t, fam_metrics[f]) for f, t in fam_tab.items()}
        for f, t in perref.items():
            t.to_csv(d / f"per_reference_{f}.csv.gz", index=False)
        grids = {"A": grid_a(), "B": grid_b(), "C": grid_c()}
        order = {f: {g["setting"]: i for i, g in enumerate(grids[f])} for f in grids}
        # ---- development grid (strict references) and selection
        selection = {}
        for f in ("A", "B", "C"):
            pr = perref[f]
            dev = pr[(pr.half == "development") & pr.ref_strict]
            g = dev.groupby(["design", "setting", "arm"])[fam_metrics[f]].mean()
            g["n_references"] = dev.groupby(["design", "setting", "arm"]).reference.nunique()
            g = g.reset_index()
            if f == "C":
                pp = fam_tab["C"]
                dv = pp[(pp.half == "development") & pp.ref_strict & (pp.arm == "same_group")]
                deg = dv.groupby("setting").degenerate_pair.mean()
                g["dev_same_group_nonterminated_share"] = g.setting.map(dv.groupby("setting").terminated.apply(lambda s: float((~s).mean())))
                g["dev_same_group_degenerate_share"] = g.setting.map(deg)
                g["degenerate_setting"] = g.dev_same_group_degenerate_share > 0.5
            g["grid_order"] = g.setting.map(order[f])
            g = g.sort_values(["design", "arm", "grid_order"])
            g.to_csv(d / f"grid_development_{f}.csv", index=False)
            cand = g[(g.design == SELECTION_DESIGN[f]) & (g.arm == "same_group")].copy()
            if f == "C":
                cand = cand[~cand.degenerate_setting]
            m = PRIMARY[f]
            cand = cand[np.isfinite(cand[m])]
            best = cand[m].max()
            top = cand[cand[m] == best]
            if f == "C" and len(top) > 1:
                top = top[top.purity_bin == top.purity_bin.max()]
            top = top.sort_values("grid_order")
            selection[f] = {"selected": top.setting.iloc[0], "submitted": SUBMITTED[f], "criterion": m,
                            "design": SELECTION_DESIGN[f], "n_tied_at_max": int((cand[m] == best).sum()),
                            "n_candidates": int(len(cand)), "dev_value_selected": float(best),
                            "dev_value_submitted": (float(g[(g.design == SELECTION_DESIGN[f]) & (g.arm == "same_group")
                                                          & (g.setting == SUBMITTED[f])][m].iloc[0])
                                                    if SUBMITTED[f] in set(g.setting) else None)}
        (d / "selection.json").write_text(json.dumps(selection, indent=1))
        # ---- report-half estimates (strict; sensitivity = strict + sensitivity-only references)
        estimates, outcome = {}, {}
        for refset in ("strict", "loose"):
            estimates[refset] = {}
            for f in ("A", "B", "C"):
                pr = perref[f]
                rep = pr[(pr.half == "report") & (pr.ref_strict if refset == "strict" else True)]
                pats_all = sorted(set(rep.ref_patient))
                pats, mult = make_patient_draws(pats_all, N_BOOT, SEED)
                sel, sub = selection[f]["selected"], SUBMITTED[f]
                fam = {}
                for design in REPORT_DESIGNS[f]:
                    for setting_role, setting in (("selected", sel), ("submitted", sub)):
                        for arm in ("same_group", "other_group"):
                            x = rep[(rep.design == design) & (rep.setting == setting) & (rep.arm == arm)]
                            for m in fam_metrics[f]:
                                xx = x[np.isfinite(x[m])]
                                if not len(xx):
                                    continue
                                reps = replicate_means(xx[m], xx.ref_patient, pats, mult)
                                fam[f"{design}|{setting_role}|{arm}|{m}"] = {
                                    "setting": setting, **summarise_reps(xx[m].mean(), reps, len(xx))}
                        # same - other contrast (per reference where both exist; else unpaired replicate means)
                        s_ = rep[(rep.design == design) & (rep.setting == setting) & (rep.arm == "same_group")]
                        o_ = rep[(rep.design == design) & (rep.setting == setting) & (rep.arm == "other_group")]
                        m = PRIMARY[f]
                        s_, o_ = s_[np.isfinite(s_[m])], o_[np.isfinite(o_[m])]
                        if len(s_) and len(o_):
                            rs = replicate_means(s_[m], s_.ref_patient, pats, mult)
                            ro = replicate_means(o_[m], o_.ref_patient, pats, mult)
                            jj = s_.merge(o_, on="reference", suffixes=("_s", "_o"))
                            fam[f"{design}|{setting_role}|same_minus_other|{m}"] = {
                                "setting": setting, "method": "difference of replicate means (same refs: "
                                f"{len(jj)} paired of {len(s_)}/{len(o_)})",
                                **summarise_reps(s_[m].mean() - o_[m].mean(), rs - ro, len(set(s_.reference) | set(o_.reference)))}
                    # paired selected - submitted, same arm and other arm
                    for arm in ("same_group", "other_group"):
                        m = PRIMARY[f]
                        xs = rep[(rep.design == design) & (rep.setting == sel) & (rep.arm == arm)][["reference", "ref_patient", m]]
                        xb = rep[(rep.design == design) & (rep.setting == sub) & (rep.arm == arm)][["reference", m]]
                        jj = xs.merge(xb, on="reference", suffixes=("_sel", "_sub"))
                        jj = jj[np.isfinite(jj[f"{m}_sel"]) & np.isfinite(jj[f"{m}_sub"])]
                        if len(jj):
                            dv = jj[f"{m}_sel"] - jj[f"{m}_sub"]
                            reps = replicate_means(dv, jj.ref_patient, pats, mult)
                            fam[f"{design}|selected_minus_submitted|{arm}|{m}"] = {
                                "selected": sel, "submitted": sub, **summarise_reps(dv.mean(), reps, len(jj))}
                estimates[refset][f] = fam
                if refset == "strict":
                    key = f"{SELECTION_DESIGN[f]}|selected_minus_submitted|same_group|{PRIMARY[f]}"
                    e = fam.get(key)
                    adopt = bool(sel != sub and e is not None and e["ci95_lo"] is not None and e["ci95_lo"] > 0)
                    outcome[f] = {"selected": sel, "submitted": sub, "rule_key": key,
                                  "adopt_selected": adopt, "figure_setting": sel if adopt else sub,
                                  "reason": ("selected == submitted" if sel == sub else
                                             "paired report-half interval lower bound > 0" if adopt else
                                             "paired report-half interval includes or is below zero")}
        # natural annotated subset (P2 pairs also P1), descriptive, selected/submitted settings
        nat = {}
        flag = p2.set_index("pair_id").also_P1 == "True"
        for f in ("A", "B", "C"):
            t = fam_tab[f]
            t = t[(t.design == "P2") & t.pair_id.map(flag).fillna(False).astype(bool)]
            for role, s_ in (("selected", selection[f]["selected"]), ("submitted", SUBMITTED[f])):
                x = t[t.setting == s_]
                for arm, xa in x.groupby("arm"):
                    nat[f"{f}|{role}|{arm}"] = {"setting": s_, "n_pairs": int(len(xa)),
                                                "n_references": int(xa.reference.nunique()),
                                                **{m: float(xa[m].mean()) for m in fam_metrics[f] if m in xa}}
        out_json = {"created": now(), "B": N_BOOT, "seed": SEED, "generator": "numpy PCG64",
                    "resampling_unit": "reference patient (report half), one draw matrix per family and reference set",
                    "ci": "percentile 2.5/97.5, nominal 95 %, unadjusted",
                    "estimates": estimates, "natural_annotated_subset": nat}
        (d / "report_estimates.json").write_text(json.dumps(out_json, indent=1, default=str))
        (d / "outcome.json").write_text(json.dumps(outcome, indent=1))
        (d / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
        cpu = pd.DataFrame([{"reference": r, **v} for r, v in done.items()])
        cpu.to_csv(d / "s1_cpu_seconds.csv", index=False)
        finish(d, "s2", {"s0_pairs": str(s0), "s1": str(s1), "c_stage": str(crun), "f1": str(f1), "f2": str(f2), "f3": str(f3),
                         "checks": checks, "selection": selection, "outcome": outcome,
                         "s1_cpu_hours": round(float(cpu.seconds.sum()) / 3600, 2)})
        print(f"[s2] done {d}", flush=True)
        return 0
    except Exception as e:  # pragma: no cover
        fail(d, "s2", repr(e), {"traceback": traceback.format_exc()})
        return 2


# ---------------------------------------------------------------- S3 representative tiles
def representative_links(T_sp, cq, cr, Xq, Xr) -> list:
    """Per cluster with >= 1 intra-cluster link: medoid, highest-mass and lowest-mass links (choice 13)."""
    nq = Xq / np.maximum(np.linalg.norm(Xq, axis=1, keepdims=True), 1e-8)
    nr = Xr / np.maximum(np.linalg.norm(Xr, axis=1, keepdims=True), 1e-8)
    ii, jj = np.nonzero(T_sp)
    out = []
    for c in sorted(set(cq.tolist()) | set(cr.tolist())):
        sel = (cq[ii] == c) & (cr[jj] == c)
        if not sel.any():
            continue
        ei, ej = ii[sel], jj[sel]
        w = T_sp[ei, ej]
        mu = np.concatenate([nq[cq == c], nr[cr == c]]).mean(axis=0)
        score = np.linalg.norm(nq[ei] - mu, axis=1) + np.linalg.norm(nr[ej] - mu, axis=1)
        picks = {"medoid": int(np.argmin(score)), "highest_mass": int(np.argmax(w)), "lowest_mass": int(np.argmin(w))}
        for kind, k in picks.items():
            out.append({"cluster": int(c), "kind": kind, "i": int(ei[k]), "j": int(ej[k]), "mass": float(w[k]),
                        "n_links": int(sel.sum()), "n_query_tiles": int((cq == c).sum()),
                        "n_ref_tiles": int((cr == c).sum())})
    return out


def stage_s3(a) -> int:
    d = new_run_dir("s3")
    try:
        from PIL import Image, ImageDraw
        import openslide
        s2 = latest_complete("s2")
        verify_complete(s2)
        s0 = latest_complete("s0_pairs")
        bind = Path(json.loads((s0 / "COMPLETE.json").read_text())["s0_bind"])
        outcome = json.loads((s2 / "outcome.json").read_text())
        setting = outcome["C"]["figure_setting"]
        _, sp, p, cl, r = setting.split("|")
        p = None if p == "None" else (float(p) if sp == "threshold" else int(p))
        r = None if r == "None" else float(r)
        _, OV = import_prosper()
        gen = ot_generator()
        p2 = pd.read_csv(s0 / "pairs_P2.csv", dtype=str)
        cand = p2[(p2.arm == "same_group") & (p2.ref_strict == "True")].drop_duplicates(["reference", "query"])
        cand = cand.assign(_k=cand.reference.map(lambda s: (len(s), s))).sort_values(["_k", "query"]).reset_index(drop=True)
        rng = np.random.Generator(np.random.PCG64(SEED))
        chosen, seen = [], set()
        for k in rng.permutation(len(cand)):
            row = cand.iloc[int(k)]
            if row.reference in seen:
                continue
            seen.add(row.reference)
            chosen.append(("annot", row["query"], row.reference))
            if len(chosen) == 3:
                break
        pairs = [(f"fig5_{panel}", q, n) for panel, q, n, _ in FIG5_PAIRS] + chosen
        fig5_res = {(q, n): res_ for _, q, n, res_ in FIG5_PAIRS}
        submitted_used = not outcome["C"]["adopt_selected"]
        legend = (f"Representative tiles per correspondence cluster. Setting {setting}"
                  f"{' (Fig. 5 pairs: published partitions, full plan, Louvain 4e-5 RMS / 5e-4 RB)' if submitted_used else ''} "
                  f"({'selected' if outcome['C']['adopt_selected'] else 'submitted'} by the pre-registered outcome rule). "
                  "Per cluster: medoid link = the intra-cluster plan link (query tile i, reference tile j) minimising "
                  "||x_i - mu|| + ||y_j - mu|| (L2-normalised UNI2 vectors, mu = their mean over all tiles of the cluster on both "
                  "slides); highest- and lowest-mass intra-cluster links. Each tile: 512 px at level 0; caption = cosine "
                  "cost, transported mass and the pathologist class where annotated (T tumour, I immune, N necrosis, "
                  "NT non-tumour, A artefact, - unannotated). Pairs: the three Fig. 5 pairs and three P2 same-group "
                  "annotated pairs with distinct references drawn with seed 123. No manual choice.")
        (d / "LEGEND.txt").write_text(legend + "\n")
        cname = {0: "-", 1: "T", 2: "I", 3: "N", 4: "NT", 5: "A"}
        rows_out = []
        TS = 112
        for tag, q, n in pairs:
            Fq, qc = load_features(q)
            Fr, rc = load_features(n)
            T, M, dist = ot_plan(gen, Fq, Fr)
            if tag.startswith("fig5"):
                saved = pickle.load(open(bound_path(bind, f"fig5_plan:{q}_{n}"), "rb"))
                rows_out.append({"pair": tag, "query": q, "reference": n, "kind": "plan_check",
                                 "mass": float(np.abs(T - np.asarray(saved["T"])).max())})
            T_sp = sparsify(T, sp, p, OV)
            if submitted_used and tag.startswith("fig5"):
                # addendum A3: the published partition (notebook construction at the published resolution) on the
                # saved Fig. 5 plan, so community ids equal the published labels (fresh plan differs by ~1e-11 and
                # relabels the RB communities); costs from the saved plan as well
                T_sp = np.asarray(saved["T"], dtype=np.float64)
                M = np.asarray(saved["M"])
                cq, cr = notebook_partition(T_sp, fig5_res[(q, n)])
                lab = pd.read_csv(bound_path(bind, f"fig5_labels:{q}_{n}"), dtype={"slide": str})
                if not (np.array_equal(cq, lab[lab.slide == q].sort_values("patch_idx").community.to_numpy()) and
                        np.array_equal(cr, lab[lab.slide == n].sort_values("patch_idx").community.to_numpy())):
                    raise RuntimeError(f"{tag}: published partition not reproduced")
                rows_out.append({"pair": tag, "query": q, "reference": n, "kind": "partition",
                                 "mass": fig5_res[(q, n)], "cluster": -1})
            else:
                G = build_graph(T_sp)
                cq, cr = cluster_graph(G, T.shape[0], T.shape[1], cl, r)
            labq = load_labels(q) if (LABELS / f"{q}.csv").exists() else {}
            labr = load_labels(n) if (LABELS / f"{n}.csv").exists() else {}
            links = representative_links(T_sp, cq, cr, Fq, Fr)
            wq, wr = openslide.OpenSlide(str(WSI).format(q)), openslide.OpenSlide(str(WSI).format(n))
            clusters = sorted(set(l["cluster"] for l in links))
            per_page = 20
            for page in range(0, max(1, len(clusters)), per_page):
                cs = clusters[page:page + per_page]
                if not cs:
                    break
                W = 80 + 3 * (2 * TS + 20)
                Hh = 30 + len(cs) * (TS + 34)
                img = Image.new("RGB", (W, Hh), "white")
                dr = ImageDraw.Draw(img)
                for x0, title in ((80, "medoid"), (80 + 2 * TS + 20, "highest mass"), (80 + 2 * (2 * TS + 20), "lowest mass")):
                    dr.text((x0, 8), f"{title} (query | reference)", fill="black")
                for ri, c in enumerate(cs):
                    y = 30 + ri * (TS + 34)
                    dr.text((4, y + TS // 2), f"c{c}", fill="black")
                    for ci, kind in enumerate(("medoid", "highest_mass", "lowest_mass")):
                        l = [x for x in links if x["cluster"] == c and x["kind"] == kind][0]
                        x0 = 80 + ci * (2 * TS + 20)
                        for si, (ws, coords, idx, lab) in enumerate(((wq, qc, l["i"], labq), (wr, rc, l["j"], labr))):
                            x, yy = coords[idx]
                            tile = ws.read_region((int(x), int(yy)), 0, (512, 512)).convert("RGB").resize((TS, TS))
                            img.paste(tile, (x0 + si * TS, y))
                            l[f"{'query' if si == 0 else 'ref'}_class"] = cname.get(lab.get((int(x), int(yy)), 0), "?")
                            l[f"{'query' if si == 0 else 'ref'}_xy"] = f"{x},{yy}"
                        l["cosine_cost"] = float(M[l["i"], l["j"]])
                        dr.text((x0, y + TS + 2), f"c={l['cosine_cost']:.3f} m={l['mass']:.1e}", fill="black")
                        dr.text((x0, y + TS + 16), f"{l['query_class']} | {l['ref_class']}", fill="black")
                img.save(d / f"montage_{tag}_{q}_vs_{n}_p{page // per_page + 1:02d}.png")
            for l in links:
                rows_out.append({"pair": tag, "query": q, "reference": n, **l})
        pd.DataFrame(rows_out).to_csv(d / "representative_links.csv", index=False)
        finish(d, "s3", {"s2": str(s2), "setting": setting, "pairs": pairs})
        print(f"[s3] done {d}", flush=True)
        return 0
    except Exception as e:  # pragma: no cover
        fail(d, "s3", repr(e), {"traceback": traceback.format_exc()})
        return 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["bind", "f1", "f2", "f3", "pairs", "s1-cmds", "s1", "c-cmds", "c", "c-finish", "s2", "s3"])
    p.add_argument("--reference")
    p.add_argument("--s0")
    p.add_argument("--run-dir")
    p.add_argument("--s1")
    p.add_argument("--only", help="s1-cmds/c-cmds: comma list of references")
    p.add_argument("--skip-c", action="store_true", help="s1: no family-C scoring (amendment 003: C is scored from saved plans)")
    p.add_argument("--cmds-name")
    p.add_argument("--c-run")
    p.add_argument("--c-prev")
    a = p.parse_args(argv)
    return {"bind": stage_bind, "f1": stage_f1, "f2": stage_f2, "f3": stage_f3, "pairs": stage_pairs,
            "s1-cmds": stage_s1_cmds, "s1": stage_s1, "c-cmds": stage_c_cmds, "c": stage_c,
            "c-finish": stage_c_finish, "s2": stage_s2, "s3": stage_s3}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
