"""R2.5 Fig. 5 (design decision 10-06, Q1 (a)): assemble the revised Fig. 5 panels from the ORIGINAL overlays plus
an H&E column, without re-clustering.

The submitted Fig. 5 panels are the composite PNGs written by the found driver notebook
(comet-analysis/src/visualize_slide_similarity.ipynb; RB pair at Louvain resolution 5e-4, RMS pairs at 4e-5),
and each replays pixel-exactly from the saved exact OT plans (plan §12.10, lead_fig5_driver_match/replay_*.json).
This script therefore only (i) authenticates each saved overlay PNG against the sha256 recorded by the replay,
(ii) reads the unaltered H&E at the same OpenSlide level (level 3, 32-fold) and size, (iii) assembles one row per
query-neighbour pair (query H&E | query overlay | neighbour H&E | neighbour overlay) with titles taken from the
slide metadata table and ranks/distances from the reviewed option-(b) recipe (itself derived from the UNI2 OT
ranked list over the 2,778-slide library), each image carrying a 2-mm scale bar derived from the slide's
recorded microns-per-pixel (panels are NOT at a common scale), (iv) archives the replay JSONs and the RB label
tables next to the figure, and (v) writes a recipe.json with every input/output hash (WSI hashes recomputed and
checked against the replay). Refuses a non-empty output directory.

Review fixes applied 10-06 ([assistant] [internal record] F12/F13/F6): scale bars; explicit error when a neighbour is
absent from the option-(b) recipe; WSI sha256 recomputed; timezone-aware timestamp; wrapped titles.

Usage: python3 scripts/r2_5_fig5_submitted_recipe_composite.py OUT_DIR
"""
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
import shutil

import numpy as np
import openslide
import pandas as pd
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 1 << 28
Image.MAX_IMAGE_PIXELS = None

H = "/PROSPER_WORKSPACE"
FIG = f"{H}/comet-feature-gen/comet-analysis/fig"
NOTEBOOK = f"{H}/comet-feature-gen/comet-analysis/src/visualize_slide_similarity.ipynb"
WSI = f"{H}/data/comet_final_HPC/{{}}.svs"
META = f"{H}/comet-new-scans/data/final_slide_list/final_scan_id_disease_map_rich_data.csv"
PROSPER_OUT = os.environ.get("PROSPER_OUT", os.path.join(os.getcwd(), "prosper_revision_output"))
REPLAY_DIR = f"{PROSPER_OUT}/review_scratch_batch2_r25_20261006/lead_fig5_driver_match"
OPTB_RECIPE = f"{PROSPER_OUT}/r2_5_fig5_render_20261006/production/recipe.json"
LEVEL = 3
SCALE_BAR_UM = 2000.0
ARCHIVE_FILES = ["rb_labels_7333_7330_5e-4.csv", "rb_community_counts.csv", "rb_lens_box_counts.json"]

# panel -> list of (query, neighbour, resolution_tag, replay json)
PANELS = {
    "A": [("16875", "16867", "4e-05", "replay_16875_16867_4e-05.json"),
          ("16875", "15155", "4e-05", "replay_16875_15155_4e-05.json")],
    "B": [("7333", "7330", "0.0005", "replay_7333_7330_0.0005.json")],
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_row(meta, scan):
    rows = meta[meta["slide_id"].astype(str) == str(scan)]
    if len(rows) != 1:
        sys.exit(f"metadata: expected exactly one row for scan {scan}, found {len(rows)}")
    return rows.iloc[0]


def title_for(meta, scan, role, rank=None, distance=None):
    r = meta_row(meta, scan)
    s = f"{role}: {r['tumor_group']} | {r['Comp.Bio.ID']} | {r['Site']} | scan {scan}"
    if rank is not None:
        s += f" | rank {rank} (OT-cosine distance {distance:.3f})"
    return s


def he_image(scan):
    sl = openslide.OpenSlide(WSI.format(scan))
    dims = sl.level_dimensions[LEVEL]
    ds = sl.level_downsamples[LEVEL]
    mpp = float(sl.properties[openslide.PROPERTY_NAME_MPP_X])
    img = sl.read_region((0, 0), LEVEL, dims).convert("RGB")
    sl.close()
    return img, dims, ds, mpp


def draw_scale_bar(ax, im, full_width_px, um_per_px_level):
    """2-mm bar; im is the thumbnail actually shown, full_width_px the level-{LEVEL} width it was scaled from."""
    scale = im.width / float(full_width_px)
    bar_px = SCALE_BAR_UM / um_per_px_level * scale
    x0, y0 = im.width * 0.04, im.height * 0.95
    ax.plot([x0, x0 + bar_px], [y0, y0], color="black", linewidth=3, solid_capstyle="butt")
    ax.text(x0, y0 - im.height * 0.015, f"{SCALE_BAR_UM/1000:.0f} mm", fontsize=7, color="black", va="bottom")


def main(out_dir):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"refusing non-empty output directory {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(META, dtype=str)
    optb = json.load(open(OPTB_RECIPE))
    rec = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": os.path.relpath(__file__, os.getcwd()),
        "script_sha256": sha256_file(__file__),
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "hostname": socket.gethostname(),
        "python": sys.version, "platform": platform.platform(),
        "versions": {"openslide": openslide.__version__, "PIL": Image.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
        "decision": "Author 10-06 Q1 (a): keep the submitted overlays (found driver, pixel-exact replay) and add an "
                    "H&E column; no re-clustering. Option-(b) regeneration is archival only.",
        "driver_notebook": {"path": NOTEBOOK, "sha256": sha256_file(NOTEBOOK)},
        "metadata": {"path": META, "sha256": sha256_file(META), "rows": int(len(meta))},
        "option_b_recipe_for_ranks": {"path": OPTB_RECIPE, "sha256": sha256_file(OPTB_RECIPE)},
        "level": LEVEL, "scale_bar_um": SCALE_BAR_UM,
        "rank_source": "UNI2 features, OT with cosine ground cost, leave-one-out over the 2,778-slide COMET library "
                       "(c2_metric_suite ranked list, via the option-(b) recipe); the query's own patient is excluded",
        "note_scale": "Each image is shown at its own level-3 size (thumbnail <=1600 px); panels are not at a common "
                      "scale - read the per-image 2-mm bar.",
        "archived": {}, "panels": {},
    }
    arch_dir = os.path.join(out_dir, "provenance")
    os.makedirs(arch_dir)
    for name in ARCHIVE_FILES + sorted({p[3] for pairs in PANELS.values() for p in pairs}):
        src = os.path.join(REPLAY_DIR, name)
        dst = os.path.join(arch_dir, name)
        shutil.copyfile(src, dst)
        rec["archived"][name] = {"source": src, "sha256": sha256_file(dst)}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for panel, pairs in PANELS.items():
        prow = []
        prec = []
        for q, n, res_tag, replay_name in pairs:
            replay = json.load(open(os.path.join(REPLAY_DIR, replay_name)))
            if replay["pair"] != [q, n]:
                sys.exit(f"replay {replay_name} pair mismatch {replay['pair']} vs {[q, n]}")
            cands = [p for p in optb["panels"][panel]["pairs"] if p["neighbour"] == n]
            if len(cands) != 1:
                sys.exit(f"option-b recipe: expected one entry for neighbour {n} in panel {panel}, found {len(cands)}")
            optb_pair = cands[0]
            if optb["panels"][panel]["query"] != q:
                sys.exit(f"option-b recipe query mismatch for panel {panel}")
            entry = {"query": q, "neighbour": n, "resolution": replay["resolution"], "networkx_replay": replay["networkx"],
                     "n_communities": replay["n_communities"], "plan_sha256": replay["plan_sha256"],
                     "rank": optb_pair["rank"], "distance": optb_pair["distance"], "slides": {}}
            imgs = {}
            for sid in (q, n):
                saved = f"{FIG}/{q}_{n}/heatmap_slide_{sid}_{res_tag}.png"
                sha = sha256_file(saved)
                r = replay["results"][sid]
                if sha != r["saved_sha256"]:
                    sys.exit(f"saved overlay {saved} sha256 {sha} != replay-recorded {r['saved_sha256']}")
                if r["n_pixels_differing_any"] != 0:
                    sys.exit(f"replay of {sid} was not pixel-exact; refusing")
                ov = Image.open(saved).convert("RGB")
                he, dims, ds, mpp = he_image(sid)
                wsi_sha = sha256_file(WSI.format(sid))
                if wsi_sha != replay["wsi_sha256"][sid]:
                    sys.exit(f"WSI {sid} sha256 {wsi_sha} != replay-recorded {replay['wsi_sha256'][sid]}")
                if ov.size != tuple(dims):
                    sys.exit(f"overlay size {ov.size} != level-{LEVEL} dims {dims} for {sid}")
                he_path = os.path.join(out_dir, f"{panel}_{q}_vs_{n}_{sid}_he_level{LEVEL}.png")
                he.save(he_path)
                imgs[sid] = (he, ov, dims[0], mpp * ds)
                entry["slides"][sid] = {"overlay_saved_png": saved, "overlay_sha256": sha, "he_png": he_path,
                                        "he_sha256": sha256_file(he_path), "wsi": WSI.format(sid),
                                        "wsi_sha256": wsi_sha, "mosaic_sha256": replay["mosaic_sha256"][sid],
                                        "level_dims": list(dims), "downsample": float(ds), "mpp_level0": mpp,
                                        "um_per_px_at_level": mpp * ds,
                                        "replay_overlay_pixels": r["overlay_region_px"], "replay_pixels_differing": r["n_pixels_differing_any"]}
            prec.append(entry)
            prow.append({"q_he": imgs[q][0], "q_ov": imgs[q][1], "n_he": imgs[n][0], "n_ov": imgs[n][1],
                         "q_geom": imgs[q][2:], "n_geom": imgs[n][2:],
                         "q_title": title_for(meta, q, "Query"),
                         "n_title": title_for(meta, n, "Neighbour", entry["rank"], entry["distance"])})
        fig, axes = plt.subplots(len(prow), 4, figsize=(26, 5.5 * len(prow)))
        axes = np.atleast_2d(axes)
        for i, row in enumerate(prow):
            cells = [(row["q_he"], row["q_title"] + "  (H&E)", row["q_geom"]),
                     (row["q_ov"], "Query: OT correspondence communities overlaid on the H&E (this pair)", row["q_geom"]),
                     (row["n_he"], row["n_title"] + "  (H&E)", row["n_geom"]),
                     (row["n_ov"], "Neighbour: OT correspondence communities overlaid on the H&E (this pair)", row["n_geom"])]
            for j, (im, t, (full_w, um_px)) in enumerate(cells):
                im = im.copy()
                im.thumbnail((1600, 1600))
                axes[i, j].imshow(im)
                axes[i, j].axis("off")
                axes[i, j].set_title(t, fontsize=7, loc="left", wrap=True)
                draw_scale_bar(axes[i, j], im, full_w, um_px)
        fig.suptitle("Colours are specific to each query-neighbour pair (one clustering per row); do not compare colours across rows. "
                     "Images are not at a common scale; each carries a 2-mm bar.",
                     fontsize=9, y=0.995)
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"Figure_5{panel}_revised.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        rec["panels"][panel] = {"query": pairs[0][0], "pairs": prec, "composite": out_png, "composite_sha256": sha256_file(out_png)}
    with open(os.path.join(out_dir, "recipe.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({p: rec["panels"][p]["composite_sha256"] for p in rec["panels"]}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
