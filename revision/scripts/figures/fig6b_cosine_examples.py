"""Figure 6B (revised, cosine): precompute the two ROI-search examples under UNI2 + cosine.

Design decision (2026-11-11): cosine for both ROI search and heat map; 2026-11-12: Fig. 6A shows the
six configurations and the dependent displays are redrawn. The submitted panel B (one EWS and one NB
query slide) used the published Virchow2[:1280] L1 hits. The same two query slides and the same
query ROIs are kept: the ROI tiles were identified by image matching of the submitted crops against
every mosaic tile of the query slide (EWS row 26, correlation 0.837 against 0.457 for the next tile;
NB row 2002, 0.880 against 0.643), and the submitted ROI1-ROI3 crops equal the published top-three hit
tiles of those rows (0.84-0.87). Only the search configuration changes.

Per example: the query ROI's hits from ``hits/uni2_cos`` (R3.5-ROI6), skipping hit slides of the
query's patient (as in the Fig. 6A aggregation); the first three remaining hits are ROI1-ROI3. The
similarity map on ROI1's slide follows Methods "ROI similarity heatmaps" with cosine distance: cosine
distance from the query ROI's UNI2 feature to every mosaic tile of the slide, assigned to the
512 x 512 patch locations each tile represents (``cluster_locs``; overlaps averaged), min-max
normalized within the pair and inverted.

Gates: query ROI feature = UNI2 bank row; ROI1 hit tile is the argmin of the cosine distances over the
slide's mosaic tiles; that distance equals the stored hit distance (1e-5); every mosaic tile of each
displayed slide aligns to a bank row by level-0 coordinate.

Needs OpenSlide: ``LD_LIBRARY_PATH=$HOME/miniconda3/envs/clam_latest/lib``.
Output: $PROSPER_OUT/figures_20261021/fig6_roi_search/panelB_cosine/ (tile PNGs, thumbnails,
heat grids, examples.json); ``fig6_roi_search.py`` renders from these files only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, prosper_out, sha256  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import r3_5_l1_maps as L  # noqa: E402

ROOT = prosper_out() / "roi_search_6cfg_20261111"
BANK = ROOT / "banks" / "uni2"
HITS = ROOT / "hits" / "uni2_cos"
SUBMITTED_HITS = Path("/PROSPER_WORKSPACE/comet-new-scans/output/"
                      "top_k_hit_slides_final_slides_final")
META_CSV = Path(L.C4_META)  # the ROI evaluation's metadata (Slide.Scan.File, Tumor.group, UID.Subject)
EXAMPLES = [  # (query scan, query mosaic row identified from the submitted crop, display group)
    ("10075", 26, "EWS"),
    ("15381", 2002, "NB"),
]
THUMB_MAX = 1600
OUT = prosper_out() / "figures_20261021" / "fig6_roi_search" / "panelB_cosine"


def bank_rows(slides, sid):
    r = slides[slides.slide_id == int(sid)].iloc[0]
    a, n = int(r.row_start), int(r.n_rows)
    with open(BANK / "features.f32", "rb") as f:
        f.seek(a * 1536 * 4)
        X = np.fromfile(f, dtype=np.float32, count=n * 1536).reshape(n, 1536)
    with open(BANK / "coords.i64", "rb") as f:
        f.seek(a * 16)
        C = np.fromfile(f, dtype=np.int64, count=n * 2).reshape(n, 2)
    return X, C


def save_png(im, path):
    """Pixels as read from the slide, without the scanner ICC profile (too large for PIL's PNG reader)."""
    im.info.pop("icc_profile", None)
    im.save(path)


def tile_image(ws, xy):
    return ws.read_region((int(xy[0]), int(xy[1])), 0, (512, 512)).convert("RGB")


def thumbnail(ws):
    W, H = ws.dimensions
    ds = max(W, H) / THUMB_MAX
    lvl = ws.get_best_level_for_downsample(ds)
    im = ws.read_region((0, 0), lvl, ws.level_dimensions[lvl]).convert("RGB")
    size = (round(W / ds), round(H / ds))
    from PIL import Image
    return im.resize(size, Image.LANCZOS), ds


def main() -> int:
    import openslide
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = pd.read_csv(BANK / "slides.csv")
    ali = pd.read_csv(REPO / "data" / "ALIdf3Ali.csv", dtype=str, usecols=["Slide.Scan.File", "Comp.Bio.ID"])
    cbid = dict(zip(ali["Slide.Scan.File"], ali["Comp.Bio.ID"]))
    meta = pd.read_csv(META_CSV, dtype=str)
    pid = dict(zip(meta["Slide.Scan.File"], meta["UID.Subject"]))
    grp = dict(zip(meta["Slide.Scan.File"], meta["Tumor.group"]))
    gates, recs, need = {}, [], set()
    plan = []
    for q, row, g in EXAMPLES:
        h = pd.read_pickle(HITS / f"top_10_hit_slides_slide_{q}.pkl")
        h = h[h.query_patch_id == row]
        assert len(h) == 10 and (np.diff(h.similarity.to_numpy()) >= 0).all()
        h = h.assign(hit_slide_id=h.hit_slide_id.astype(str))
        keep = h[h.hit_slide_id.map(pid) != pid[q]].head(3)
        dropped = h[h.hit_slide_id.map(pid) == pid[q]]
        assert len(keep) == 3
        plan.append((q, row, g, keep, dropped))
        need |= {q, *keep.hit_slide_id}
    print(f"loading {L.META_PKL.name} ...", flush=True)
    raw = pd.read_pickle(L.META_PKL)
    mtiles = {e[0].split("/")[-1].replace(".svs", ""): e[1] for e in raw if e[1] is not None and
              e[0].split("/")[-1].replace(".svs", "") in need}
    del raw
    print(f"metadata {time.time() - t0:.0f}s", flush=True)
    for q, row, g, keep, dropped in plan:
        d = OUT / f"{g}_{q}"
        d.mkdir(exist_ok=True)
        Xq, Cq = bank_rows(slides, q)
        qv = Xq[row]
        wq = openslide.OpenSlide(str(L.WSI).format(q))
        save_png(tile_image(wq, Cq[row]), d / "query_roi.png")
        th, ds_q = thumbnail(wq)
        save_png(th, d / "query_thumb.png")
        hits = []
        for j, (_, r) in enumerate(keep.iterrows(), 1):
            s = r.hit_slide_id
            Xs, Cs = bank_rows(slides, s)
            ws = openslide.OpenSlide(str(L.WSI).format(s))
            save_png(tile_image(ws, Cs[int(r.hit_patch_id)]), d / f"roi{j}.png")
            dist = 1.0 - (Xs @ qv) / (np.linalg.norm(Xs, axis=1) * np.linalg.norm(qv))
            hits.append(dict(rank_after_exclusion=j, hit_slide=s, hit_slide_id=cbid.get(s), group=grp.get(s),
                             hit_row=int(r.hit_patch_id), stored_distance=float(r.similarity),
                             recomputed_distance=float(dist[int(r.hit_patch_id)]),
                             hit_xy=[int(v) for v in Cs[int(r.hit_patch_id)]]))
            # ROI1 is the displayed map; if ROI1 is from another group, the first same-group ROI's
            # map is also written (alternative display, not selected).
            first_same = j > 1 and grp.get(s) == g and all(h["group"] != g for h in hits[:-1])
            if j == 1 or first_same:
                tag = "roi1" if j == 1 else "same_group"
                gates[f"{g}:{tag}_is_argmin"] = int(np.argmin(dist)) == int(r.hit_patch_id)
                gates[f"{g}:{tag}_distance_matches"] = abs(float(dist[int(r.hit_patch_id)]) - float(r.similarity)) <= 1e-5
                pos = {(int(x), int(y)): i for i, (x, y) in enumerate(Cs)}
                tiles = mtiles[s]
                aligned = [(t, pos.get((int(t["wsi_loc"][0]), int(t["wsi_loc"][1])))) for t in tiles]
                gates[f"{g}:{tag}_slide_tiles_aligned"] = all(i is not None for _, i in aligned) and len(aligned) == len(Cs)
                patch = {}
                for t, i in aligned:
                    for xy in t.get("cluster_locs", [t["wsi_loc"]]):
                        patch.setdefault((int(xy[0]), int(xy[1])), []).append(i)
                locs = np.array(list(patch), dtype=np.int64)
                v = np.array([dist[o].mean() for o in patch.values()])
                heat = L.normalise_invert(v)
                rt, ds_r = thumbnail(ws)
                save_png(rt, d / f"{tag}_thumb.png")
                np.savez_compressed(d / f"{tag}_heat.npz", locs=locs, heat=heat, downsample=ds_r,
                                    hit_xy=np.array(hits[-1]["hit_xy"]), patch=512, rank=j)
        recs.append(dict(group=g, query=q, query_slide_id=cbid.get(q), query_patient_excluded=True,
                         query_row=row, query_xy=[int(v) for v in Cq[row]], query_thumb_downsample=ds_q,
                         same_patient_hits_skipped=dropped.hit_slide_id.tolist(), hits=hits,
                         submitted_top3=pd.read_pickle(SUBMITTED_HITS / f"top_10_hit_slides_slide_{q}.pkl")
                         .query("query_patch_id == @row").head(3)[["hit_slide_id", "hit_patch_id"]].astype(str).values.tolist()))
    rec = dict(gates=gates, passed=all(gates.values()), examples=recs,
               inputs={str(p): sha256(p) for p in [BANK / "slides.csv", *(HITS / f"top_10_hit_slides_slide_{q}.pkl" for q, _, _ in EXAMPLES)]},
               bank_features_sha256=json.loads((BANK / "bank.json").read_text())["features_sha256"],
               script_sha256=sha256(Path(__file__)), seconds=round(time.time() - t0, 1),
               created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (OUT / "examples.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(rec["gates"], indent=1))
    for r in recs:
        print(r["group"], r["query_slide_id"], "skipped", r["same_patient_hits_skipped"],
              [(h["hit_slide_id"], h["group"], round(h["stored_distance"], 4)) for h in r["hits"]])
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
