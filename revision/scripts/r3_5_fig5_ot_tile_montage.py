#!/usr/bin/env python3
"""R3.5-iii tile montage for the revised Fig. 5 (design decision Q7(b) / batch-3 E3; F delta map §E, adapted to the
DISPLAYED partition): query | neighbour tile pairs linked by the saved exact optimal-transport plans, sampled within
each displayed community in proportion to transported mass. Qualitative only; no annotation class, no purity number.

Partition = exactly the one shown in the revised Fig. 5 (author 10-06 Q1(a): submitted overlays, replayed pixel-exactly
by `review_scratch_batch2_r25_20261006/lead_fig5_driver_match/nb_recipe_replay.py` from the found driver notebook):
Louvain (networkx 3.5, seed 0) on the full bipartite graph of nonzero plan entries, nodes inserted as "A_i"/"B_j" in
np.nonzero order, resolution 4e-5 for the RMS pairs (16875->15155, 16875->16867) and 5e-4 for the RB pair (7333->7330);
community id = 1-based enumeration order; colour = matplotlib Set1[(cid-1) % 9]. The script refuses to run unless the
networkx version is 3.5, the community count equals the replay JSON, and (RB) the labels equal the archived
`rb_labels_7333_7330_5e-4.csv` exactly.

Sampling: within each community, 6 intra-community plan edges (i, j) without replacement with probability T[i, j]
(all edges if fewer), rng = np.random.default_rng(np.random.PCG64(SEED)). Every sampled pair is shown; no curation.
Tiles: 512 x 512 level-0 windows at the mosaic tile's `wsi_loc` (COMET_final/mosaic/<scan>.pkl, the mosaic the plan
rows/columns index), read with OpenSlide and downsampled to 160 px. Outputs: one PNG per pair, an edge CSV, label CSVs
per pair, metrics.json with all input hashes, sha256.txt, COMPLETE.
"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

H = "/PROSPER_WORKSPACE"
OT_DIR = f"{H}/comet-feature-gen/comet-uni2/output/similarity_pairwise_ot"
MOSAIC = "/PROSPER_DEPT/core_operations/DNB_Core_Public/COMET_portal_files/COMET_final/mosaic/{}.pkl"
WSI = f"{H}/data/comet_final_HPC/{{}}.svs"
META = f"{H}/comet-new-scans/data/final_slide_list/final_scan_id_disease_map_rich_data.csv"
PAIRS = [("A1", "16875", "15155", 4e-5, "replay_16875_15155_4e-05.json"),
         ("A2", "16875", "16867", 4e-5, "replay_16875_16867_4e-05.json"),
         ("B", "7333", "7330", 5e-4, "replay_7333_7330_0.0005.json")]
TILE = 512
DISPLAY = 160
PER_COMMUNITY = 6
SEED = 20261008
REQUIRED_NX = "3.5"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def displayed_partition(T, res):
    """Verbatim cells 12 of the driver notebook (as replayed in nb_recipe_replay.py)."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    if nx.__version__ != REQUIRED_NX:
        raise RuntimeError(f"networkx {nx.__version__} != {REQUIRED_NX}; the displayed partition is bound to {REQUIRED_NX}")
    G = nx.Graph()
    for i, j in zip(*np.nonzero(T)):
        G.add_edge(f"A_{i}", f"B_{j}", weight=float(T[i, j]))
    comms = louvain_communities(G, weight="weight", resolution=res, seed=0)
    n_a, n_b = T.shape
    lab_a = np.zeros(n_a, dtype=int)
    lab_b = np.zeros(n_b, dtype=int)
    for cid, nodes in enumerate(comms, start=1):
        for u in nodes:
            side, k = u.split("_")
            (lab_a if side == "A" else lab_b)[int(k)] = cid
    if (lab_a == 0).any() or (lab_b == 0).any():
        raise RuntimeError("some mosaic tiles received no community (unexpected: every tile has plan mass)")
    return lab_a, lab_b, len(comms)


def main():
    t0 = time.time()
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        raise RuntimeError("PROSPER_OUT must be exported")
    replay_dir = f"{out_root}/review_scratch_batch2_r25_20261006/lead_fig5_driver_match"
    rb_labels_path = f"{out_root}/r2_5_fig5_submitted_recipe_20261006/provenance/rb_labels_7333_7330_5e-4.csv"
    OUT = os.path.join(out_root, "r3_5_fig5_ot_tile_montage_20261008_v2")
    os.makedirs(OUT, exist_ok=False)

    import openslide
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from PIL import Image
    set1 = colormaps["Set1"].colors

    meta = pd.read_csv(META, dtype=str)
    meta_sha = sha256_file(META)
    rng = np.random.default_rng(np.random.PCG64(SEED))
    rec = dict(script=os.path.basename(__file__), script_sha256=sha256_file(os.path.abspath(__file__)), seed=SEED,
               per_community=PER_COMMUNITY, display_px=DISPLAY, meta=dict(path=META, sha256=meta_sha), pairs={})
    edges_out = []
    slides = {}
    wsi_sha = {}

    def slide_of(s):
        if s not in slides:
            slides[s] = openslide.OpenSlide(WSI.format(s))
            wsi_sha[s] = sha256_file(WSI.format(s))
        return slides[s]

    def read_tile(s, x, y):
        im = slide_of(s).read_region((int(x), int(y)), 0, (TILE, TILE)).convert("RGB")
        return im.resize((DISPLAY, DISPLAY), Image.LANCZOS)

    def label_of(s):
        r = meta[meta["slide_id"] == s]
        if len(r) != 1:
            raise RuntimeError(f"{s}: {len(r)} metadata rows")
        return r.iloc[0].to_dict()

    for panel, q, n, res, replay_name in PAIRS:
        plan_path = f"{OT_DIR}/ot_{q}_vs_{n}.pkl"
        d = pickle.load(open(plan_path, "rb"))
        if d["slide_a"] != q or d["slide_b"] != n:
            raise RuntimeError(f"{plan_path}: slide ids {d['slide_a']},{d['slide_b']}")
        T = np.asarray(d["T"], dtype=np.float64)
        M = np.asarray(d["M"], dtype=np.float64)
        if T.shape != M.shape or not np.isfinite(T).all() or (T < 0).any():
            raise RuntimeError(f"{plan_path}: malformed T/M")
        replay = json.load(open(f"{replay_dir}/{replay_name}"))
        if replay["plan_sha256"] != sha256_file(plan_path) or abs(replay["resolution"] - res) > 1e-12:
            raise RuntimeError(f"{panel}: plan hash / resolution differ from the displayed-figure replay")
        mos_q = pickle.load(open(MOSAIC.format(q), "rb"))
        mos_n = pickle.load(open(MOSAIC.format(n), "rb"))
        for s, m in ((q, mos_q), (n, mos_n)):
            if sha256_file(MOSAIC.format(s)) != replay["mosaic_sha256"][s]:
                raise RuntimeError(f"{s}: mosaic hash differs from replay")
        if len(mos_q) != T.shape[0] or len(mos_n) != T.shape[1]:
            raise RuntimeError(f"{panel}: mosaic sizes {len(mos_q)},{len(mos_n)} != plan {T.shape}")
        xy_q = np.array([m["wsi_loc"] for m in mos_q], dtype=int)
        xy_n = np.array([m["wsi_loc"] for m in mos_n], dtype=int)

        lab_a, lab_b, n_comm = displayed_partition(T, res)
        if n_comm != replay["n_communities"]:
            raise RuntimeError(f"{panel}: {n_comm} communities != replay {replay['n_communities']}")
        if panel == "B":
            rb = pd.read_csv(rb_labels_path, dtype={"slide": str})
            for s, lab in ((q, lab_a), (n, lab_b)):
                r = rb[rb.slide == s].sort_values("patch_idx")
                if len(r) != len(lab) or not (r.community.to_numpy() == lab).all():
                    raise RuntimeError(f"{panel}: labels differ from archived rb_labels for {s}")
        pd.DataFrame(dict(slide=[q] * len(lab_a) + [n] * len(lab_b), patch_idx=list(range(len(lab_a))) + list(range(len(lab_b))),
                          community=np.concatenate([lab_a, lab_b]))).to_csv(f"{OUT}/labels_{q}_{n}.csv", index=False)

        ii, jj = np.nonzero(T)
        intra = lab_a[ii] == lab_b[jj]
        comm_rows = []
        sampled = {}
        for cid in range(1, n_comm + 1):
            sel = intra & (lab_a[ii] == cid)
            ei, ej = ii[sel], jj[sel]
            w = T[ei, ej]
            k = min(PER_COMMUNITY, len(ei))
            pick = rng.choice(len(ei), size=k, replace=False, p=w / w.sum()) if k else np.array([], dtype=int)
            sampled[cid] = [(int(ei[p]), int(ej[p])) for p in pick]
            comm_rows.append(dict(community=cid, tiles_query=int((lab_a == cid).sum()), tiles_neighbour=int((lab_b == cid).sum()),
                                  intra_edges=int(sel.sum()), intra_mass=float(w.sum()), n_sampled=k))
            for r, (i, j) in enumerate(sampled[cid], start=1):
                edges_out.append(dict(panel=panel, query=q, neighbour=n, community=cid, sample_rank=r, i=i, j=j, T=float(T[i, j]),
                                      M_cosine_cost=float(M[i, j]), q_x=int(xy_q[i, 0]), q_y=int(xy_q[i, 1]),
                                      n_x=int(xy_n[j, 0]), n_y=int(xy_n[j, 1])))
        cross_mass = float(T[ii[~intra], jj[~intra]].sum())

        # render: rows = communities, columns = sampled pairs (query | neighbour)
        mq, mn = label_of(q), label_of(n)
        fig_w = 1.3 + PER_COMMUNITY * 2.15
        fig_h = 0.9 + n_comm * 1.12
        fig, axes = plt.subplots(n_comm, PER_COMMUNITY * 2, figsize=(fig_w, fig_h), squeeze=False)
        plt.subplots_adjust(left=0.08, right=0.995, top=1 - 0.55 / fig_h, bottom=0.05 / fig_h, wspace=0.04, hspace=0.12)
        for cid in range(1, n_comm + 1):
            col = set1[(cid - 1) % len(set1)]
            for c in range(PER_COMMUNITY):
                axq, axn = axes[cid - 1, 2 * c], axes[cid - 1, 2 * c + 1]
                for ax in (axq, axn):
                    ax.set_xticks([]); ax.set_yticks([])
                if c < len(sampled[cid]):
                    i, j = sampled[cid][c]
                    axq.imshow(read_tile(q, *xy_q[i])); axn.imshow(read_tile(n, *xy_n[j]))
                    for ax in (axq, axn):
                        for sp in ax.spines.values():
                            sp.set_edgecolor(col); sp.set_linewidth(3)
                    axq.set_title(f"q{i}", fontsize=5, pad=1); axn.set_title(f"n{j}  cos-cost {M[i, j]:.2f}", fontsize=5, pad=1)
                else:
                    axq.axis("off"); axn.axis("off")
            cr = comm_rows[cid - 1]
            axes[cid - 1, 0].set_ylabel(f"community {cid}\n{cr['tiles_query']} / {cr['tiles_neighbour']} tiles\nmass {cr['intra_mass']:.2f}",
                                        fontsize=6, color="black", rotation=0, ha="right", va="center", labelpad=6)
        mpp_q = slide_of(q).properties.get("openslide.mpp-x"); mpp_n = slide_of(n).properties.get("openslide.mpp-x")
        um_q = float(mpp_q) * TILE if mpp_q else float("nan"); um_n = float(mpp_n) * TILE if mpp_n else float("nan")
        fig.suptitle(f"Panel {panel}: query {q} ({mq['Diagnosis']}) | neighbour {n} "
                     f"({mn['Diagnosis']}); OT distance {d['distance']:.4f}; Louvain resolution {res:g}, "
                     f"{n_comm} communities as displayed in Fig. 5\nEach pair = one plan edge (query tile | neighbour tile), sampled "
                     f"∝ transported mass within its community; all sampled pairs shown. Tile = 512 px ≈ {um_q:.0f} µm (query) / {um_n:.0f} µm (neighbour).",
                     fontsize=7)
        png = f"{OUT}/montage_{panel}_{q}_vs_{n}.png"
        fig.savefig(png, dpi=220)
        plt.close(fig)
        rec["pairs"][panel] = dict(query=q, neighbour=n, resolution=res, n_communities=n_comm, plan=dict(path=plan_path, sha256=replay["plan_sha256"],
                                   shape=list(T.shape), nnz=int(len(ii)), distance=float(d["distance"])),
                                   mosaic_sha256=replay["mosaic_sha256"], replay_json=replay_name, cross_community_mass=cross_mass,
                                   communities=comm_rows, png=os.path.basename(png),
                                   mpp_x={q: mpp_q, n: mpp_n}, tile_um={q: um_q, n: um_n})
        print(panel, q, n, "communities", n_comm, [c["n_sampled"] for c in comm_rows], f"cross-mass {cross_mass:.4f}", flush=True)

    pd.DataFrame(edges_out).to_csv(f"{OUT}/sampled_edges.csv", index=False)
    rec["wsi_sha256"] = wsi_sha
    rec["n_pairs_total"] = len(edges_out)
    rec["networkx"] = __import__("networkx").__version__
    rec["library_versions"] = {m: __import__(m).__version__ for m in ("numpy", "pandas", "networkx", "matplotlib", "openslide", "PIL")}
    rec["python"] = sys.version
    rec["runtime_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(f"{OUT}/metrics.json", "w"), indent=1)
    with open(f"{OUT}/README.md", "w") as fh:
        fh.write("# Fig. 5 OT tile montage (R3.5-iii)\n\n" + __doc__ + f"\nPairs shown: {len(edges_out)}. Runtime {rec['runtime_s']} s.\n")
    with open(f"{OUT}/sha256.txt", "w") as fh:
        for name in sorted(os.listdir(OUT)):
            if name not in ("sha256.txt", "COMPLETE"):
                fh.write(f"{sha256_file(os.path.join(OUT, name))}  {name}\n")
    open(f"{OUT}/COMPLETE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print("wrote", OUT, f"{rec['runtime_s']} s")


if __name__ == "__main__":
    sys.exit(main())
