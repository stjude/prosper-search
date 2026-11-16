#!/usr/bin/env python3
"""R2.5 option (b): re-render Fig. 5 OT correspondence-cluster overlays with an unoverlaid H&E panel per slide.

Design decision F1 (2026-10-06): keep the published pairs (query 16875 with neighbours 16867 and 15155; query
7333 with 7330), correct the 15155 patient ID from the metadata CSV, and print the ACTUAL rank of each
neighbour in the frozen accepted UNI2 OT-cosine ranking (16867 = rank 1, 15155 = rank 10), never "top 5".

Method (CPU only; no prosper import; re-implemented from prosper-search/prosper/visualization/
ot_visualization.py 1f664155..., functions sparsify_transport_matrix(mutual_topk, verbatim logic) and
cluster_patches_from_transport(louvain branch; the reference DEFAULT is connected components, and the
shared/grey distinction and size ordering below are additions of this script)): saved exact OT plan T (uniform
marginals, cosine cost) -> optional mutual top-k sparsification (--k 0 = none) -> optional global mass-quantile
threshold (default off) -> weighted bipartite graph over ALL mosaic tiles of both slides -> networkx
louvain_communities(weight, resolution, seed 0) -> each tile coloured by its community (palette index = community
size rank WITHIN the pair; colours are pair-specific) on both slides at OpenSlide level L (512-px level-0 tile ->
512/downsample block), alpha-composited over the level-L H&E. Communities containing tiles of only one slide
carry no correspondence and are drawn grey. Every parameter set tried is appended to trials.jsonl (with the
script hash); recipe.json is per trial. Nothing is hand-typed into titles (group/site/ID come from the metadata
CSV, ranks from the frozen ranked list).

Structural facts found during analyst trials (2026-10-06): the saved exact plans are basic feasible solutions,
i.e. spanning trees (nnz = n_a + n_b - 1, one connected component, row nnz <= 9 here). Hence mutual top-k is
a no-op for k >= 10 and FRAGMENTS the tree at the reference k = 5 (56 shared communities on the 15155 pair),
and Louvain on a tree needs a small resolution to yield few communities; at that point the partition is where
the tree can be cut almost for free under Louvain's size penalty, not a unique set of "correspondence regions".
Design decision (2026-10-06): full plan (--k 0), resolution 0.0001 (resolutions 1.0 down to 0.00005 were tried;
0.0001 chosen for legibility; seeds 0-9 give 4-6 / 7-8 / 3 communities, recorded by --seed-sensitivity);
disclosure of the parameter choice and of the differences from the submitted figure is documented with the figure,
the recipe in Methods/legend.

Colouring (--fill nearest; design decision 2026-10-06): the transport plan is defined only on the ~15 %
mosaic tiles, while the submitted figure colours the whole tissue because the shipped renderer painted each
representative's label onto its spatial k-means group within its colour cluster (`cluster_locs`, never saved;
the published mosaics predate seed pinning in mosaic.py, so those groups are unrecoverable). Here every tissue
tile of the regenerated 512-px tissue grid (scripts/r2_5_fig5_tissue_grid.py, which PASSes only if the saved
mosaic is a subset of that grid, and whose input hashes are re-checked here) takes the label of the NEAREST
saved mosaic tile (Euclidean distance on level-0 coordinates). This differs from the original grouping for
roughly a third to a half of tissue tiles (independent measurement 52-67 % agreement on regenerated mosaics).
Ties (17-22 % of tissue tiles have >= 2 equidistant centres; 8-17 % have tied centres with DIFFERENT labels) are
resolved by the majority label among the tied centres, then the lowest mosaic index among the majority labels.
Most label-conflicting ties are 1:1 splits, so roughly 7-16 % of tissue tiles are still coloured by index order;
the counts of ties, label-conflicting ties and tiles whose colour would change under the reverse (highest-index)
rule are recorded per slide. --fill none paints only the mosaic tiles.

Reference-figure parameters were never recorded (Methods L245 has an empty "(, )"), so re-rendered colours and
partitions differ from the submitted figure. Outputs: per-slide H&E and overlay PNGs, per-pair composite rows
(query H&E / query overlay / neighbour H&E / neighbour overlay, so colours are never read across pairs),
cluster table, recipe.json with input and output hashes/versions. Refuses a non-empty output directory unless
--append-trial.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("PROSPER_OUT", os.path.join(R, "prosper_revision_output"))
H = "/PROSPER_WORKSPACE"
OT_DIR = f"{H}/comet-feature-gen/comet-uni2/output/similarity_pairwise_ot"
COORD_DIR = f"{H}/comet-feature-gen/comet-uni2/data/tmp_final"
WSI_DIR = f"{H}/data/comet_final_HPC"
META = f"{H}/comet-new-scans/data/final_slide_list/final_scan_id_disease_map_rich_data.csv"
RANKED = os.path.join(OUT, "c2_metric_suite", "ranked_lists_uni2_ot_cosine.csv.gz")
PANELS = {"A": ("16875", ["16867", "15155"]), "B": ("7333", ["7330"])}
TILE = 512
GREY = (160, 160, 160)
# Okabe-Ito (first 7 entries colour-blind-safe) + extras; render refuses more shared communities than SAFE_PALETTE
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0),
           (204, 121, 167), (128, 0, 128), (0, 128, 128), (255, 0, 0), (0, 200, 0), (100, 100, 255)]
SAFE_PALETTE = 7


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def load_meta():
    m = pd.read_csv(META, dtype=str, keep_default_na=False)
    # 110 slide_id values are duplicated in this file (93 with differing Comp.Bio.ID); bind per slide, not globally
    return m, dict(path=META, sha256=sha256_file(META), rows=int(len(m)),
                   rows_with_duplicated_slide_id=int(m["slide_id"].duplicated(keep=False).sum()),
                   n_duplicated_slide_id_values=int(m.loc[m["slide_id"].duplicated(keep=False), "slide_id"].nunique()))


def meta_row(meta, scan):
    rows = meta[meta["slide_id"] == scan]
    if len(rows) != 1:
        raise ValueError(f"slide {scan}: {len(rows)} metadata rows, need exactly one")
    return rows.iloc[0]


def load_ranks():
    with gzip.open(RANKED, "rt") as f:
        r = pd.read_csv(f, dtype={"query": str, "neighbour": str}, keep_default_na=False, float_precision="round_trip")
    return r, sha256_file(RANKED)


def load_plan(a, b):
    p = os.path.join(OT_DIR, f"ot_{a}_vs_{b}.pkl")
    with open(p, "rb") as f:
        d = pickle.load(f)
    if d["slide_a"] != a or d["slide_b"] != b:
        raise ValueError(f"{p}: slide ids {d['slide_a']},{d['slide_b']} != {a},{b}")
    T = np.asarray(d["T"], dtype=np.float64)
    if T.ndim != 2 or not np.isfinite(T).all() or (T < 0).any():
        raise ValueError(f"{p}: malformed T")
    return dict(path=p, sha256=sha256_file(p), shape=list(T.shape), nnz=int((T > 0).sum()), mass=float(T.sum()),
                distance=float(d["distance"]), M_shape=list(np.shape(d["M"]))), T


def load_coords(scan, expected_n):
    p = os.path.join(COORD_DIR, f"{scan}.pkl")
    with open(p, "rb") as f:
        feats, locs = pickle.load(f)
    if len(locs) != expected_n or np.shape(feats)[0] != expected_n:
        raise ValueError(f"{p}: {len(locs)} coordinates / {np.shape(feats)[0]} features != {expected_n} plan rows")
    xy = np.array([[int(l["wsi_loc"][0]), int(l["wsi_loc"][1])] for l in locs], dtype=np.int64)
    if (xy % TILE).any():
        raise ValueError(f"{p}: coordinates are not on the {TILE}-px level-0 grid")
    return dict(path=p, sha256=sha256_file(p), n_tiles=int(expected_n), feature_dim=int(np.shape(feats)[1])), xy


def mutual_topk(T, k):
    """Verbatim logic of the reference sparsify_transport_matrix(method='mutual_topk')."""
    n_a, n_b = T.shape
    row_topk = np.argpartition(-T, kth=min(k, n_b - 1), axis=1)[:, :k]
    row_mask = np.zeros_like(T, dtype=bool)
    row_mask[np.arange(n_a)[:, None], row_topk] = True
    col_topk = np.argpartition(-T, kth=min(k, n_a - 1), axis=0)[:k, :]
    col_mask = np.zeros_like(T, dtype=bool)
    col_mask[col_topk, np.arange(n_b)[None, :]] = True
    return np.where(row_mask & col_mask, T, 0.0)


def cluster(T, k, quantile, resolution, seed=0):
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    T_sp = mutual_topk(T, k) if k > 0 else np.array(T, dtype=np.float64, copy=True)
    n_edges_topk = int((T_sp > 0).sum())
    if quantile > 0:
        pos = T_sp[T_sp > 0]
        thr = float(np.quantile(pos, quantile)) if pos.size else 0.0
        T_sp = np.where(T_sp >= thr, T_sp, 0.0)
    n_a, n_b = T.shape
    G = nx.Graph()
    G.add_nodes_from(range(n_a), slide="A")
    G.add_nodes_from(range(n_a, n_a + n_b), slide="B")
    ii, jj = np.nonzero(T_sp)
    G.add_weighted_edges_from((int(i), int(n_a + j), float(T_sp[i, j])) for i, j in zip(ii, jj))
    comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    lab_a = np.full(n_a, -1, dtype=np.int64)
    lab_b = np.full(n_b, -1, dtype=np.int64)
    rows = []
    cid = 0
    for nodes in sorted(comms, key=lambda s: (-len(s), min(s))):
        ia = [u for u in nodes if u < n_a]
        ib = [u - n_a for u in nodes if u >= n_a]
        shared = bool(ia) and bool(ib)
        if shared:
            cid += 1
            lab_a[ia] = cid
            lab_b[ib] = cid
        mass = float(T_sp[np.ix_(ia, ib)].sum()) if shared else 0.0
        rows.append(dict(cluster_id=cid if shared else 0, shared=shared, tiles_a=len(ia), tiles_b=len(ib),
                         sparsified_mass=mass))
    tab = pd.DataFrame(rows)
    stats = dict(n_tiles_a=n_a, n_tiles_b=n_b, nonzero_plan_entries=int((T > 0).sum()), edges_after_mutual_topk=n_edges_topk,
                 edges_after_quantile=int((T_sp > 0).sum()), retained_mass=float(T_sp.sum()),
                 n_communities=len(comms), n_shared_communities=int(cid),
                 tiles_a_in_shared=int((lab_a > 0).sum()), tiles_b_in_shared=int((lab_b > 0).sum()))
    return lab_a, lab_b, tab, stats


def load_tissue_grid(tissue_dir, scan, xy_mosaic, coords_sha256, wsi_sha256):
    """Regenerated 512-px tissue grid for `scan`; saved mosaic must be a subset (re-checked) and the grid must have
    been built from the SAME coordinate pickle and WSI (hash equality), by a PASS run of the tissue-grid script."""
    p = os.path.join(tissue_dir, f"{scan}_tissue_grid.json")
    with open(p) as f:
        g = json.load(f)
    with open(os.path.join(tissue_dir, "summary.json")) as f:
        summ = json.load(f)
    if g.get("status") != "PASS" or summ.get("status") != "PASS":
        raise ValueError(f"{p}: tissue-grid status {g.get('status')}/{summ.get('status')} is not PASS")
    if g.get("coords_sha256") != coords_sha256 or g.get("wsi_sha256") != wsi_sha256:
        raise ValueError(f"{p}: tissue grid was built from different coordinate/WSI bytes than the current inputs")
    tissue = np.array(g["tissue_tiles"], dtype=np.int64).reshape(-1, 2)
    if (tissue % TILE).any():
        raise ValueError(f"{p}: tissue tiles are not on the {TILE}-px grid")
    tissue_set = {tuple(t) for t in tissue.tolist()}
    if len(tissue_set) != len(tissue):
        raise ValueError(f"{p}: duplicate tissue tiles")
    missing = [tuple(t) for t in xy_mosaic.tolist() if tuple(t) not in tissue_set]
    if missing:
        raise ValueError(f"{p}: {len(missing)} saved mosaic tiles are not in the regenerated tissue grid, e.g. {missing[:3]}")
    return dict(path=p, sha256=sha256_file(p), summary_sha256=sha256_file(os.path.join(tissue_dir, "summary.json")),
                tissue_grid_script_sha256=summ.get("script_sha256"), n_tissue_tiles=int(len(tissue))), tissue


def nearest_fill(xy_mosaic, labels, tissue):
    """Label every tissue tile with the label of its nearest mosaic-tile centre (exact integer squared Euclidean
    distance, level-0 px). Ties (several equidistant centres): majority label among the tied centres, then lowest
    mosaic index. Returns labels, nearest index (lowest tied), and tie statistics incl. the number of tiles whose
    label would differ under the reverse rule (highest index), as a sensitivity measure."""
    m = xy_mosaic.astype(np.int64)
    n = len(tissue)
    out = np.empty(n, dtype=np.int64)
    out_rev = np.empty(n, dtype=np.int64)
    nearest = np.empty(n, dtype=np.int64)
    n_ties = 0
    n_conflict = 0
    for s in range(0, n, 1024):
        t = tissue[s:s + 1024].astype(np.int64)
        d2 = ((t[:, None, :] - m[None, :, :]) ** 2).sum(axis=2)
        dmin = d2.min(axis=1)
        tied = d2 == dmin[:, None]
        for r in range(len(t)):
            idx = np.flatnonzero(tied[r])
            nearest[s + r] = idx[0]
            if len(idx) == 1:
                out[s + r] = out_rev[s + r] = labels[idx[0]]
                continue
            n_ties += 1
            labs = labels[idx]
            if len(set(labs.tolist())) > 1:
                n_conflict += 1
            vals, counts = np.unique(labs, return_counts=True)
            top = vals[counts == counts.max()]
            # majority label; if the majority is itself tied, the label of the lowest (resp. highest) tied index
            sel = labs[np.isin(labs, top)]  # restrict the index fallback to the majority labels only
            out[s + r] = top[0] if len(top) == 1 else sel[0]
            out_rev[s + r] = top[0] if len(top) == 1 else sel[-1]
    stats = dict(tie_rule="majority label among equidistant centres, then lowest mosaic index",
                 n_tissue_tiles=int(n), n_ties=int(n_ties), n_ties_with_conflicting_labels=int(n_conflict),
                 n_tiles_changed_under_reverse_rule=int((out != out_rev).sum()),
                 n_tiles_changed_vs_lowest_index_rule=int((out != labels[nearest]).sum()))
    return out, nearest, stats


def render(scan, xy, labels, level, alpha, out_png_he, out_png_ov, fill="none", tissue_dir=None,
           coords_sha256=None, wsi_sha256=None):
    import openslide
    from PIL import Image
    path = os.path.join(WSI_DIR, f"{scan}.svs")
    n_shared = int(labels.max()) if len(labels) else 0
    if n_shared > SAFE_PALETTE:
        raise ValueError(f"{scan}: {n_shared} shared communities exceed the {SAFE_PALETTE} colour-blind-safe palette entries")
    sl = openslide.OpenSlide(path)
    actual_ds = float(sl.level_downsamples[level])
    ds = 4 ** level  # nominal, as the reference (int(patch_size / 4**slide_level)); actual Aperio downsamples are ~16.002
    if abs(actual_ds - ds) > 0.05 * ds:
        raise ValueError(f"{path}: level {level} downsample {actual_ds} far from nominal {ds}")
    block = TILE // ds
    W, Hh = sl.level_dimensions[level]
    he = sl.read_region((0, 0), level, (W, Hh)).convert("RGBA")
    ov = np.zeros((Hh, W, 4), dtype=np.uint8)
    fill_info = dict(fill=fill)
    if fill == "nearest":
        ginfo, tissue = load_tissue_grid(tissue_dir, scan, xy, coords_sha256, wsi_sha256)
        paint_lab, nearest, tie_stats = nearest_fill(xy, labels, tissue)
        paint_xy = tissue
        # the mosaic tiles themselves are in the tissue grid and are their own unique nearest centre (distance 0)
        mosaic_index = {tuple(t): i for i, t in enumerate(tissue.tolist())}
        if not all(nearest[mosaic_index[tuple(t)]] == i and paint_lab[mosaic_index[tuple(t)]] == labels[i]
                   for i, t in enumerate(xy.tolist())):
            raise ValueError(f"{scan}: a mosaic tile was not assigned to itself in the nearest-centre fill")
        counts = np.bincount(nearest, minlength=len(xy))
        fill_info.update(tissue_grid=ginfo, n_tissue_tiles=int(len(tissue)), n_mosaic_tiles=int(len(xy)),
                         n_tissue_in_shared=int((paint_lab > 0).sum()), ties=tie_stats,
                         tissue_tiles_per_mosaic_centre=dict(min=int(counts.min()), median=float(np.median(counts)),
                                                             max=int(counts.max())))
    elif fill == "none":
        paint_xy, paint_lab = xy, labels
    else:
        raise ValueError(f"unknown fill {fill}")
    for (x, y), lab in zip(paint_xy, paint_lab):
        c = GREY if lab <= 0 else PALETTE[(lab - 1) % len(PALETTE)]
        x0, y0 = x // ds, y // ds
        ov[y0:y0 + block, x0:x0 + block, :3] = c
        ov[y0:y0 + block, x0:x0 + block, 3] = alpha
    comp = Image.alpha_composite(he, Image.fromarray(ov))
    he.convert("RGB").save(out_png_he)
    comp.convert("RGB").save(out_png_ov)
    info = dict(wsi=path, wsi_target=os.path.realpath(path), wsi_sha256=wsi_sha256 or sha256_file(path), level=level,
                downsample_nominal=ds, downsample_actual=actual_ds, level_dims=[W, Hh], level0_dims=list(sl.level_dimensions[0]),
                block_px=block, alpha=alpha, vendor=sl.properties.get("openslide.vendor"), mpp_x=sl.properties.get("openslide.mpp-x"),
                he_sha256=sha256_file(out_png_he), overlay_sha256=sha256_file(out_png_ov), **fill_info)
    sl.close()
    return info


def title_for(meta, scan, role, rank=None, distance=None):
    r = meta_row(meta, scan)
    s = f"{role}: {r['tumor_group']} | {r['Comp.Bio.ID']} | {r['Site']} | scan {scan}"
    if rank is not None:
        s += f" | rank {rank} (OT-cosine distance {distance:.3f})"
    return s


def composite(pair_rows, out_path):
    """One row per (query, neighbour) pair: query H&E | query overlay | neighbour H&E | neighbour overlay.
    Colours are pair-specific (independent Louvain run per pair) and must never be read across rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    n = len(pair_rows)
    fig, axes = plt.subplots(n, 4, figsize=(26, 5.5 * n))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(pair_rows):
        panels = [(row["q_he"], row["q_title"] + "  (H&E)"), (row["q_ov"], "Query: OT correspondence clusters (this pair)"),
                  (row["n_he"], row["n_title"] + "  (H&E)"), (row["n_ov"], "Neighbour: OT correspondence clusters (this pair)")]
        for j, (pth, t) in enumerate(panels):
            im = Image.open(pth)
            im.thumbnail((1600, 1600))
            axes[i, j].imshow(im)
            axes[i, j].axis("off")
            axes[i, j].set_title(t, fontsize=8, loc="left")
    fig.suptitle("Colours are specific to each query-neighbour pair (one clustering per row); do not compare colours across rows",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return sha256_file(out_path)


def seed_sensitivity(T, k, quantile, resolution, seeds):
    """Number of shared communities for each Louvain seed (F4 disclosure); seed 0 is the rendered one."""
    return {int(sd): cluster(T, k, quantile, resolution, seed=sd)[3]["n_shared_communities"] for sd in seeds}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(OUT, f"r2_5_fig5_render_{dt.date.today():%Y%m%d}"))
    ap.add_argument("--k", type=int, default=5, help="mutual top-k (reference cluster function default 5); 0 = no sparsification")
    ap.add_argument("--quantile", type=float, default=0.0, help="global mass quantile on surviving edges; 0 = off")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--alpha", type=int, default=100)
    ap.add_argument("--fill", choices=["none", "nearest"], default="none",
                    help="none = colour mosaic tiles only; nearest = every tissue tile takes its nearest mosaic centre's label")
    ap.add_argument("--tissue-dir", default=None, help="output dir of scripts/r2_5_fig5_tissue_grid.py (required for --fill nearest)")
    ap.add_argument("--trial", default="trial_001", help="sub-directory for this parameter set")
    ap.add_argument("--append-trial", action="store_true", help="allow adding a trial to an existing out-dir")
    ap.add_argument("--skip-render", action="store_true", help="clustering + tables only (parameter exploration)")
    ap.add_argument("--seed-sensitivity", type=int, default=0, metavar="N",
                    help="also run Louvain with seeds 0..N-1 and record the shared-community counts (rendering uses seed 0)")
    a = ap.parse_args(argv)
    if a.k < 0:
        ap.error("--k must be >= 0 (0 = no sparsification)")
    if a.fill == "nearest" and not (a.tissue_dir and os.path.isdir(a.tissue_dir)):
        sys.exit("--fill nearest requires --tissue-dir pointing to a PASS run of scripts/r2_5_fig5_tissue_grid.py")
    if os.path.exists(a.out_dir) and os.listdir(a.out_dir) and not a.append_trial:
        sys.exit(f"refusing to write into non-empty {a.out_dir} (use --append-trial)")
    tdir = os.path.join(a.out_dir, a.trial)
    if os.path.exists(tdir) and os.listdir(tdir):
        sys.exit(f"refusing to overwrite trial {tdir}")
    os.makedirs(tdir, exist_ok=True)

    import networkx, openslide, PIL, matplotlib
    meta, meta_info = load_meta()
    ranks, ranks_sha = load_ranks()
    recipe = dict(generated=dt.datetime.now().isoformat(), script=os.path.relpath(__file__, R),
                  script_sha256=sha256_file(__file__), params=dict(k=a.k, quantile=a.quantile, resolution=a.resolution,
                                                                   level=a.level, alpha=a.alpha, louvain_seed=0,
                                                                   fill=a.fill, tissue_dir=a.tissue_dir),
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                                networkx=networkx.__version__, openslide=openslide.__version__, PIL=PIL.__version__,
                                matplotlib=matplotlib.__version__),
                  metadata=meta_info, ranked_list=dict(path=RANKED, sha256=ranks_sha),
                  panels={}, decision="F1 (author, 2026-10-06): keep published pairs, honest ranks, corrected IDs")
    cluster_tables = []
    wsi_sha = {}
    for panel, (q, nbrs) in PANELS.items():
        qr = ranks[ranks["query"] == q].sort_values("rank")
        prow = dict(query=q, query_title=title_for(meta, q, "Query"), pairs=[])
        for nb in nbrs:
            hit = qr[qr["neighbour"] == nb]
            if len(hit) != 1:
                raise ValueError(f"neighbour {nb} not found exactly once in the depth-{len(qr)} ranked list of {q}")
            rank, dist = int(hit["rank"].iloc[0]), float(hit["distance"].iloc[0])
            plan, T = load_plan(q, nb)
            if abs(plan["distance"] - dist) > 1e-6:
                raise ValueError(f"{q} vs {nb}: plan distance {plan['distance']} != ranked-list distance {dist}")
            ca, xa = load_coords(q, T.shape[0])
            cb, xb = load_coords(nb, T.shape[1])
            la, lb, tab, stats = cluster(T, a.k, a.quantile, a.resolution)
            if a.seed_sensitivity > 0:
                stats["shared_communities_by_seed"] = seed_sensitivity(T, a.k, a.quantile, a.resolution, range(a.seed_sensitivity))
            tab.insert(0, "pair", f"{q}_vs_{nb}")
            cluster_tables.append(tab)
            pair = dict(neighbour=nb, rank=rank, distance=dist, same_site=bool(meta_row(meta, q)["Site"] == meta_row(meta, nb)["Site"]),
                        neighbour_title=title_for(meta, nb, "Neighbour", rank, dist), plan=plan, coords={q: ca, nb: cb},
                        clustering=stats)
            if not a.skip_render:
                files = {}
                for scan, xy, lab in ((q, xa, la), (nb, xb, lb)):
                    he = os.path.join(tdir, f"{panel}_{q}_vs_{nb}_{scan}_HE.png")
                    ov = os.path.join(tdir, f"{panel}_{q}_vs_{nb}_{scan}_overlay.png")
                    csha = (ca if scan == q else cb)["sha256"]
                    if scan not in wsi_sha:
                        wsi_sha[scan] = sha256_file(os.path.join(WSI_DIR, f"{scan}.svs"))
                    files[scan] = dict(he=he, overlay=ov, **render(scan, xy, lab, a.level, a.alpha, he, ov, a.fill, a.tissue_dir,
                                                                   coords_sha256=csha, wsi_sha256=wsi_sha[scan]))
                pair["render"] = files
            prow["pairs"].append(pair)
        recipe["panels"][panel] = prow
    pd.concat(cluster_tables).to_csv(os.path.join(tdir, "cluster_table.csv"), index=False)
    if not a.skip_render:
        # One row per pair, neighbours in rank order (F1: never pair a neighbour with another pair's query overlay)
        for panel, prow in recipe["panels"].items():
            q = prow["query"]
            rows = []
            for p in sorted(prow["pairs"], key=lambda p: p["rank"]):
                nb = p["neighbour"]
                rows.append(dict(q_he=p["render"][q]["he"], q_ov=p["render"][q]["overlay"], q_title=prow["query_title"],
                                 n_he=p["render"][nb]["he"], n_ov=p["render"][nb]["overlay"], n_title=p["neighbour_title"]))
            out_png = os.path.join(tdir, f"Figure_5{panel}_revised.png")
            prow["composite"] = dict(path=out_png, sha256=composite(rows, out_png), layout="one row per pair: query H&E, query overlay, neighbour H&E, neighbour overlay")
    recipe["resolution_scan_note"] = "all parameter sets tried are in trials.jsonl next to this trial directory; resolution chosen for legibility (author decision 2026-10-06)"
    with open(os.path.join(tdir, "recipe.json"), "w") as f:
        json.dump(recipe, f, indent=2, default=str)
    with open(os.path.join(a.out_dir, "trials.jsonl"), "a") as f:
        f.write(json.dumps(dict(trial=a.trial, generated=recipe["generated"], script_sha256=recipe["script_sha256"],
                                params=recipe["params"], rendered=not a.skip_render,
                                clustering={f"{p}:{x['neighbour']}": x["clustering"] for p, pr in recipe["panels"].items()
                                            for x in pr["pairs"]})) + "\n")
    for p, pr in recipe["panels"].items():
        for x in pr["pairs"]:
            s = x["clustering"]
            print(f"{p} {pr['query']} vs {x['neighbour']} rank {x['rank']} dist {x['distance']:.4f} same_site={x['same_site']}: "
                  f"{s['n_shared_communities']} shared communities, tiles in shared {s['tiles_a_in_shared']}/{s['n_tiles_a']} "
                  f"and {s['tiles_b_in_shared']}/{s['n_tiles_b']}, edges {s['edges_after_quantile']}, mass {s['retained_mass']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
