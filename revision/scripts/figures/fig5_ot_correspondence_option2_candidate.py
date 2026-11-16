"""Fig. 5 CANDIDATE (R3.5, design decision 2026-11-13 "option 2"; 19:42 CST "the match cost map should be there for
all A B comparison"): the accepted layout of fig5_ot_correspondence.py plus a MATCHED-COST MAP column for BOTH slides of
every pair (query and retrieved neighbour), i.e. the quantity that the R3.5-L1/L2/L3 localisation evaluation grades
([internal record] E1-RU7 candidate002). The first draft (18:20 CST, neighbour side only) is kept in candidates_20261113_option2/.

Draft for internal reading only. Writes to a candidate directory; the artwork of record
($PROSPER_OUT/figures_20261021/fig5_ot_correspondence/Figure_5_revised.png) is not touched.

Content of the new columns (one per slide, per pair)
  * Plan: the SAME saved exact OT plan that produced the displayed communities
    (comet-uni2/output/similarity_pairwise_ot/ot_<query>_vs_<neighbour>.pkl; SHA-256 must equal the recipe's
    plan_sha256; its stored distance must equal the recipe distance; uniform marginals; ground cost M = cosine
    distance between UNI2 tile features, re-checked against the feature pickles on a random sample).
  * Tile value: mass-weighted mean ground cost of everything transported through the tile; neighbour tile j:
    c_j = sum_i T_ij M_ij / sum_i T_ij (r3_5_l1_maps.b_tile_values(..., "meancost")); query tile i, symmetrically:
    c_i = sum_j T_ij M_ij / sum_j T_ij. Lower = better match. The query's map depends on the neighbour, so the
    same query slide (16875) has a different map in rows A0 and A1.
  * Patch projection: every 512-px level-0 patch of a slide takes the value of the mosaic tile that
    represents it (`cluster_locs` from the bound mosaic metadata, s0_bind/run_001/meta_tiles.pkl; each patch has
    exactly one representative, so this equals r3_5_l1_maps.project(..., "mean")).
  * Display: min-max normalised JOINTLY over the two slides of the pair and inverted, h = 1 - (c - min)/(max - min)
    (r3_5_l1_maps.normalise_invert applied to the concatenated pair; the same colour means the same cost on both
    sides; per-slide scaling would only rescale each map, AUROC-type metrics are unaffected), painted with the Fig. 6 ROI-heatmap style
    (jet, alpha 0.55) over the recipe's level-3 H&E render using the recipe's actual level downsample.
    Red = lowest matched cost (best-matched neighbour tissue), blue = highest. It is NOT a tumour segmentation.

Everything else (H&E renders, community overlays, identities, rank/distance line, 2-mm bars, legibility check,
manifest, provenance) follows the accepted renderer; the parked question of removing the three distances
(item 4) is left unchanged.

Usage (from the repository root):
    export PROSPER_OUT="$PWD/prosper_revision_output"
    /HOME/miniconda3/bin/python3 scripts/figures/fig5_ot_correspondence_option2_candidate.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig5_ot_correspondence as base  # noqa: E402
from _common import out_dir, prosper_out, sha256, write_provenance  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

CANDIDATE = "candidates_20261113_option2_both_sides"
FIGURE = f"{base.FIGURE}/{CANDIDATE}"
STEM = "Figure_5_candidate_option2_both_sides"

H = Path("/PROSPER_WORKSPACE")
OT_DIR = H / "comet-feature-gen/comet-uni2/output/similarity_pairwise_ot"
FEATURES = H / "comet-feature-gen/comet-uni2/data/tmp_final"
META_TILES = prosper_out() / "r3_5_localisation_maps_20261017/s0_bind/run_001/meta_tiles.pkl"
TILE = 512
HEAT_CMAP, HEAT_ALPHA = "jet", 0.55          # as scripts/figures/fig6_roi_search.py (ROI heatmap)
N_COLS = 6
GAP_PAIR = 0.12                              # slightly tighter than the accepted 0.16 to fit six columns
COLORBAR_H = 0.30                            # extra bottom strip for the matched-cost colour bar (inches)
SUBHEADERS = ["H&E", "Correspondence\ncommunities", "Matched-cost\nmap", "H&E", "Correspondence\ncommunities", "Matched-cost\nmap"]
HEADER_H = 0.48                              # two-line sub-headers in the narrower five-column cells
BAR_TOL = (0.93, 1.05)                       # bars are ~40-50 px long at 600 dpi here; antialiased ends cost ~2 px


def fail(msg: str) -> None:
    sys.exit(f"fig5_option2_candidate: {msg}")


def load_plan(q: str, n: str, expected_sha: str, expected_distance: float):
    p = OT_DIR / f"ot_{q}_vs_{n}.pkl"
    got = sha256(p)
    if got != expected_sha:
        fail(f"{p}: sha256 {got} != recipe plan_sha256 {expected_sha}")
    d = pickle.load(open(p, "rb"))
    if (str(d["slide_a"]), str(d["slide_b"])) != (q, n):
        fail(f"{p}: slide ids {d['slide_a']},{d['slide_b']} != {q},{n}")
    T = np.asarray(d["T"], dtype=np.float64)
    M = np.asarray(d["M"], dtype=np.float64)
    if T.shape != M.shape or T.ndim != 2 or not np.isfinite(T).all() or (T < 0).any():
        fail(f"{p}: malformed T/M {T.shape} {M.shape}")
    if abs(float(d["distance"]) - expected_distance) > 1e-9:
        fail(f"{p}: stored distance {d['distance']} != recipe {expected_distance}")
    return p, got, T, M, float(d["distance"])


def load_features(sid: str):
    feats, coords = pd.read_pickle(FEATURES / f"{sid}.pkl")
    xy = [tuple(int(v) for v in c["wsi_loc"]) for c in coords]
    return np.asarray(feats, dtype=np.float64), xy


def fidelity(T: np.ndarray, M: np.ndarray, Fq: np.ndarray, Fn: np.ndarray, distance: float, rng) -> dict:
    """Independent checks that the saved plan is the pipeline's exact OT with cosine ground cost and uniform
    marginals on these feature rows."""
    nq, nn = T.shape
    if (nq, nn) != (len(Fq), len(Fn)):
        fail(f"plan shape {T.shape} != feature rows {len(Fq)}x{len(Fn)}")
    ii = rng.integers(0, nq, 4000)
    jj = rng.integers(0, nn, 4000)
    cos = 1.0 - np.einsum("ij,ij->i", Fq[ii], Fn[jj]) / (np.linalg.norm(Fq[ii], axis=1) * np.linalg.norm(Fn[jj], axis=1))
    out = dict(n_query_tiles=int(nq), n_neighbour_tiles=int(nn),
               max_abs_M_minus_cosine_sample4000=float(np.abs(M[ii, jj] - cos).max()),
               row_marginal_max_abs_dev=float(np.abs(T.sum(1) - 1.0 / nq).max()),
               col_marginal_max_abs_dev=float(np.abs(T.sum(0) - 1.0 / nn).max()),
               sum_TM_minus_distance=float((T * M).sum() - distance),
               nnz=int((T > 0).sum()), nnz_over_n_plus_m_minus_1=float((T > 0).sum() / (nq + nn - 1)),
               neighbour_tiles_linked=int((T.sum(0) > 0).sum()))
    ok = (out["max_abs_M_minus_cosine_sample4000"] < 1e-5 and out["row_marginal_max_abs_dev"] < 1e-9
          and out["col_marginal_max_abs_dev"] < 1e-9 and abs(out["sum_TM_minus_distance"]) < 1e-6
          and out["neighbour_tiles_linked"] == nn)
    out["pass"] = bool(ok)
    if not ok:
        fail(f"plan fidelity check failed: {json.dumps(out)}")
    return out


def matched_cost(T: np.ndarray, M: np.ndarray, side: str) -> np.ndarray:
    """Mass-weighted mean ground cost per tile. side='neighbour' is r3_5_l1_maps.b_tile_values(T, M, 'meancost')
    for a full plan (every column linked); side='query' is the symmetric row-wise quantity."""
    ax = 0 if side == "neighbour" else 1
    return (T * M).sum(axis=ax) / T.sum(axis=ax)


def normalise_invert(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return 1.0 - (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def patches_for(sid: str, coords_xy: list, meta_tiles: dict):
    """(patch xy array, owner tile index per patch): each 512-px patch -> the mosaic tile representing it."""
    tiles = meta_tiles.get(sid)
    if tiles is None:
        fail(f"{sid}: not in bound mosaic metadata")
    if len(tiles) != len(coords_xy) or any(tuple(t["wsi_loc"]) != c for t, c in zip(tiles, coords_xy)):
        fail(f"{sid}: mosaic metadata order/content differs from the feature pickle")
    xy, owner = [], []
    for k, t in enumerate(tiles):
        for p in t["cluster_locs"]:
            xy.append((int(p[0]), int(p[1])))
            owner.append(k)
    if len(set(xy)) != len(xy):
        fail(f"{sid}: a patch has more than one representative tile (projection would need 'mean' over owners)")
    return np.asarray(xy, dtype=np.int64), np.asarray(owner, dtype=np.int64)


def paint(he: Image.Image, xy: np.ndarray, h: np.ndarray, ds: float, *, cmap=HEAT_CMAP) -> Image.Image:
    W, Hh = he.size
    ov = np.zeros((Hh, W, 4), dtype=float)
    rgba = plt.get_cmap(cmap)(h)
    for (x, y), c in zip(xy, rgba):
        x0, y0 = int(round(x / ds)), int(round(y / ds))
        x1, y1 = max(x0 + 1, int(round((x + TILE) / ds))), max(y0 + 1, int(round((y + TILE) / ds)))
        if x1 > W or y1 > Hh:
            fail(f"patch ({x},{y}) falls outside the level-3 render {W}x{Hh}")
        ov[y0:y1, x0:x1, :3] = c[:3]
        ov[y0:y1, x0:x1, 3] = HEAT_ALPHA
    ov8 = Image.fromarray((ov * 255).round().astype(np.uint8))
    return Image.alpha_composite(he.convert("RGBA"), ov8).convert("RGB")


def main() -> None:
    root = prosper_out()
    v2_dir = root / base.V2_DIR_NAME
    v2 = json.load(open(v2_dir / "recipe.json"))
    inputs = [v2_dir / "recipe.json", META_TILES]
    meta_path = v2["metadata"]["path"]
    base.check_sha(meta_path, v2["metadata"]["sha256"], "identity metadata table")
    meta = pd.read_csv(meta_path, dtype=str)
    inputs.append(Path(meta_path))
    meta_tiles = pickle.load(open(META_TILES, "rb"))
    d = out_dir(FIGURE)
    rng = np.random.default_rng(0)

    def meta_row(scan: str) -> pd.Series:
        rows = meta[meta["slide_id"].astype(str) == scan]
        if len(rows) != 1:
            fail(f"metadata: expected one row for scan {scan}, found {len(rows)}")
        r = rows.iloc[0]
        if str(r["Comp.Bio.ID"]) != base.EXPECTED_IDS[scan]:
            fail(f"scan {scan}: slide ID {r['Comp.Bio.ID']} != accepted {base.EXPECTED_IDS[scan]}")
        return r

    rows, summary = [], {}
    for panel, idx in base.ROWS:
        e = v2["panels"][panel]["pairs"][idx]
        exp = base.EXPECTED[(panel, idx)]
        q, n = str(e["query"]), str(e["neighbour"])
        if (q, n, int(e["rank"])) != (exp["query"], exp["neighbour"], exp["rank"]):
            fail(f"pair {panel}{idx}: recipe identities differ from accepted")
        if abs(float(e["distance"]) - exp["distance"]) > 5e-4 or float(e["resolution"]) != exp["resolution"]:
            fail(f"pair {panel}{idx}: distance/resolution differ from the accepted values")
        # ---- the new columns: single-pair matched-cost map on BOTH slides, jointly normalised ----
        plan_path, plan_sha, T, M, dist = load_plan(q, n, e["plan_sha256"], float(e["distance"]))
        Fq, xy_q = load_features(q)
        Fn, xy_n = load_features(n)
        inputs += [plan_path, FEATURES / f"{q}.pkl", FEATURES / f"{n}.pkl"]
        fid = fidelity(T, M, Fq, Fn, dist, rng)
        side = {}
        for sid, role, xy in ((q, "query", xy_q), (n, "neighbour", xy_n)):
            c_tile = matched_cost(T, M, role)
            pxy, owner = patches_for(sid, xy, meta_tiles)
            side[role] = dict(sid=sid, c_tile=c_tile, pxy=pxy, owner=owner, c_patch=c_tile[owner])
        lo = min(v["c_patch"].min() for v in side.values())
        hi = max(v["c_patch"].max() for v in side.values())
        for v in side.values():
            v["h"] = 1.0 - (v["c_patch"] - lo) / (hi - lo)
        tag = f"{panel}{idx}_{q}_vs_{n}"
        summary[tag] = dict(query=q, neighbour=n, rank=int(e["rank"]), distance=dist, plan=str(plan_path),
                            plan_sha256=plan_sha, fidelity=fid, joint_norm_min=float(lo), joint_norm_max=float(hi),
                            sides={})
        cells = []
        for sid, role in ((q, "query"), (n, "neighbour")):
            s = e["slides"][sid]
            he_path = base.resolve_he(v2_dir, s["he_png"])
            base.check_sha(str(he_path), s["he_sha256"], f"H&E {sid}")
            base.check_sha(s["overlay_saved_png"], s["overlay_sha256"], f"overlay {sid}")
            inputs += [he_path, Path(s["overlay_saved_png"])]
            he = Image.open(he_path).convert("RGB")
            ov = Image.open(s["overlay_saved_png"]).convert("RGB")
            if he.size != tuple(s["level_dims"]) or ov.size != tuple(s["level_dims"]):
                fail(f"image size mismatch for {sid}")
            label = base.slide_label(meta_row(sid), e["rank"] if role == "neighbour" else None,
                                     float(e["distance"]) if role == "neighbour" else None)
            um_px = float(s["um_per_px_at_level"])
            cells.append(dict(im=he, kind="he", slide=sid, role=role, src=str(he_path), sha=s["he_sha256"],
                              um_px=um_px, label=label))
            cells.append(dict(im=ov, kind="overlay", slide=sid, role=role, src=s["overlay_saved_png"],
                              sha=s["overlay_sha256"], um_px=um_px, label=label))
            v = side[role]
            mc = paint(he, v["pxy"], v["h"], float(s["downsample"]))
            mc_path = d / f"{tag}_{sid}_{role}_matched_cost_overlay.png"
            mc.save(mc_path)
            val_path = d / f"{tag}_{sid}_{role}_matched_cost_patches.csv"
            pd.DataFrame(dict(wsi_loc_x=v["pxy"][:, 0], wsi_loc_y=v["pxy"][:, 1], owner_tile=v["owner"],
                              matched_cost=v["c_patch"], display_value=v["h"])).to_csv(val_path, index=False)
            mc_sha = sha256(mc_path)
            summary[tag]["sides"][role] = dict(slide=sid, n_tiles=int(len(v["c_tile"])), n_patches=int(len(v["pxy"])),
                                               tile_cost_min=float(v["c_tile"].min()),
                                               tile_cost_median=float(np.median(v["c_tile"])),
                                               tile_cost_max=float(v["c_tile"].max()),
                                               overlay_png=str(mc_path), overlay_sha256=mc_sha,
                                               patch_values_csv=str(val_path))
            cells.append(dict(im=mc, kind="matched_cost", slide=sid, role=role, src=str(mc_path), sha=mc_sha,
                              um_px=um_px, label=label))
        rows.append(dict(panel=panel, idx=idx, pair=e, cells=cells))

    render(rows, summary, inputs, d)


def render(rows, summary, inputs, d, *, figure=FIGURE, stem=STEM, absolute_cost=False):
    # ---- geometry: five image columns on the same 180-mm page ----
    B = base
    half = N_COLS // 2
    cell_w = (B.WIDTH_IN - B.MARGIN_L - B.MARGIN_R - (N_COLS - 2) * B.GAP_X - GAP_PAIR) / N_COLS
    col_x, x = [], B.MARGIN_L
    for c in range(N_COLS):
        col_x.append(x)
        x += cell_w + (GAP_PAIR if c == half - 1 else B.GAP_X)
    # images are height-limited in the accepted layout; keep rows proportionally shorter so the narrower cells
    # are used without changing the aspect of any image
    row_h = {k: v * cell_w / ((B.WIDTH_IN - B.MARGIN_L - B.MARGIN_R - 2 * B.GAP_X - B.GAP_PAIR) / 4.0)
             for k, v in B.ROW_H.items()}
    fig_h = B.TOP_PAD + HEADER_H + B.BOTTOM_PAD + COLORBAR_H
    prev = None
    for r in rows:
        if prev is not None:
            fig_h += B.PANEL_GAP if r["panel"] != prev else B.ROW_GAP
        fig_h += B.LABEL_H + row_h[r["panel"]]
        prev = r["panel"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(B.WIDTH_IN, fig_h))
    fx, fy = (lambda v: v / B.WIDTH_IN), (lambda v: v / fig_h)
    texts, bars, manifest = [], [], []

    y = fig_h - B.TOP_PAD
    for (c0, c1, title) in ((0, half - 1, "Query slide"), (half, N_COLS - 1, "Retrieved neighbor")):
        xm = (col_x[c0] + col_x[c1] + cell_w) / 2
        t = fig.text(fx(xm), fy(y), title, fontsize=B.FONT_HEADER, fontweight="bold", ha="center", va="top")
        texts.append((t, "header", (col_x[c0], col_x[c1] + cell_w, y - 0.16, y)))
        ln = fig.add_axes([fx(col_x[c0]), fy(y - 0.17), fx(col_x[c1] + cell_w - col_x[c0]), 0.0005])
        ln.set_facecolor("black"); ln.set_xticks([]); ln.set_yticks([])
        for sp in ln.spines.values():
            sp.set_visible(False)
    for c, sub in enumerate(SUBHEADERS):
        if absolute_cost:
            sub = sub.replace("communities", "clusters")
        t = fig.text(fx(col_x[c] + cell_w / 2), fy(y - 0.20), sub, fontsize=B.FONT_SUBHEADER, ha="center", va="top",
                     linespacing=1.15)
        texts.append((t, "subheader", (col_x[c], col_x[c] + cell_w, y - HEADER_H, y - 0.18)))
    y -= HEADER_H

    prev = None
    for ri, r in enumerate(rows):
        panel = r["panel"]
        if prev is not None:
            y -= B.PANEL_GAP if panel != prev else B.ROW_GAP
        if panel != prev:
            t = fig.text(fx(0.03), fy(y + 0.01), panel, fontsize=B.FONT_PANEL, fontweight="bold", ha="left", va="top")
            texts.append((t, "panel_letter", None))
        prev = panel
        rh = row_h[panel]
        for (c0, c1, cell) in ((0, half - 1, r["cells"][0]), (half, N_COLS - 1, r["cells"][half])):
            t = fig.text(fx(col_x[c0]), fy(y - 0.005), cell["label"], fontsize=B.FONT_LABEL, ha="left", va="top",
                         linespacing=1.25)
            texts.append((t, "label", (col_x[c0], col_x[c1] + cell_w, y - B.LABEL_H, y)))
        y_img_top = y - B.LABEL_H
        for ci, cell in enumerate(r["cells"]):
            im = cell["im"]
            full_w, full_h = im.size
            sc_in = min(cell_w / full_w, rh / full_h)
            disp_w, disp_h = full_w * sc_in, full_h * sc_in
            px_w, px_h = int(round(disp_w * B.DPI)), int(round(disp_h * B.DPI))
            small = im.resize((px_w, px_h), Image.LANCZOS)
            x0 = col_x[ci] + (cell_w - disp_w) / 2
            y0 = y_img_top - rh + (rh - disp_h) / 2
            ax = fig.add_axes([fx(x0), fy(y0), fx(disp_w), fy(disp_h)])
            ax.imshow(small, interpolation="lanczos")
            ax.set_xlim(0, px_w); ax.set_ylim(px_h, 0); ax.axis("off")
            bar_px = B.SCALE_BAR_UM / cell["um_px"] * (px_w / full_w)
            bx0, by0 = px_w * 0.04, px_h * 0.955
            ax.plot([bx0, bx0 + bar_px], [by0, by0], color="black", linewidth=1.3, solid_capstyle="butt")
            lt = ax.text(bx0, by0 - px_h * 0.015, "2 mm", fontsize=B.FONT_BAR, ha="left", va="bottom",
                         bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5))
            texts.append((lt, "scale_label", (x0, x0 + disp_w, y0, y0 + disp_h)))
            bar_in = bar_px / px_w * disp_w
            bars.append(dict(row=ri, col=ci, slide=cell["slide"], kind=cell["kind"], ax=ax, bx0=bx0,
                             bx1=bx0 + bar_px, by0=by0, bar_mm=bar_in * 25.4))
            manifest.append(dict(panel=panel, row=ri, col=ci, role=cell["role"], kind=cell["kind"],
                                 slide_scan=cell["slide"], slide_id=B.EXPECTED_IDS[cell["slide"]],
                                 label=cell["label"], source=cell["src"], sha256=cell["sha"],
                                 level_px=f"{full_w}x{full_h}", um_per_px_level=cell["um_px"],
                                 displayed_mm=f"{disp_w*25.4:.1f}x{disp_h*25.4:.1f}",
                                 scale_bar_mm_on_page=round(bar_in * 25.4, 2),
                                 rank=r["pair"]["rank"] if cell["role"] == "neighbour" else "",
                                 ot_distance=round(float(r["pair"]["distance"]), 3) if cell["role"] == "neighbour" else "",
                                 louvain_resolution=r["pair"]["resolution"]))
        y = y_img_top - rh

    # ---- colour bar for the matched-cost column (bottom strip, under column 4) ----
    cb_w = cell_w * (2.0 if absolute_cost else 1.2)
    cb_xc = (col_x[0] + col_x[-1] + cell_w) / 2
    cax = fig.add_axes([fx(cb_xc - cb_w / 2), fy(B.BOTTOM_PAD + 0.12), fx(cb_w), fy(0.06)])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 1),
                                       cmap=f"{HEAT_CMAP}_r" if absolute_cost else HEAT_CMAP), cax=cax,
                      orientation="horizontal")
    if absolute_cost:
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cb.set_ticklabels(["0", "0.25", "0.50", "0.75", "1"])
    else:
        cb.set_ticks([0, 1]); cb.set_ticklabels(["highest", "lowest"])
    cb.ax.tick_params(labelsize=B.FONT_BAR, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)
    title = ("Mean matched cosine cost (lower = closer match)" if absolute_cost else
             "Matched cost (min\u2013max within pair, both slides)")
    ct = fig.text(fx(cb_xc), fy(B.BOTTOM_PAD + 0.20), title,
                  fontsize=B.FONT_BAR, ha="center", va="bottom")
    texts.append((ct, "colorbar_title", (cb_xc - 1.5 * cell_w, cb_xc + 1.5 * cell_w, B.BOTTOM_PAD, B.BOTTOM_PAD + COLORBAR_H)))
    tick_halfwidth = cb_w / 2 + 0.06 if absolute_cost else cell_w
    for lab in cb.ax.get_xticklabels():
        texts.append((lab, "colorbar_tick", (cb_xc - tick_halfwidth, cb_xc + tick_halfwidth, 0.0, B.BOTTOM_PAD + COLORBAR_H)))

    # ---- save ----
    png, pdf = d / f"{stem}.png", d / f"{stem}.pdf"
    fig.savefig(png, dpi=B.DPI, facecolor="white")
    fig.savefig(pdf, dpi=B.DPI, facecolor="white", metadata={"CreationDate": None})
    with Image.open(png) as im_png:
        rgb = Image.new("RGB", im_png.size, (255, 255, 255))
        rgba = im_png.convert("RGBA")
        rgb.paste(rgba, mask=rgba.split()[3])
        rgb.save(png, dpi=(B.DPI, B.DPI))

    # ---- legibility / geometry check (as the accepted renderer) ----
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fdpi = fig.dpi
    check = dict(width_mm=B.WIDTH_MM, height_mm=round(fig_h * 25.4, 1), dpi=B.DPI, min_font_pt_required=B.MIN_FONT_PT,
                 texts=[], scale_bars=[], overlaps=[], all_fonts_ok=True, all_texts_within_cells=True)
    boxes = []
    for art, kind, bounds in texts:
        bb = art.get_window_extent(renderer)
        x0, x1, y0, y1 = (float(v) for v in (bb.x0 / fdpi, bb.x1 / fdpi, bb.y0 / fdpi, bb.y1 / fdpi))
        ok_font = bool(art.get_fontsize() >= B.MIN_FONT_PT)
        within = True
        if bounds is not None:
            b0, b1, c0, c1 = bounds
            within = bool(x0 >= b0 - 0.01 and x1 <= b1 + 0.01 and y0 >= c0 - 0.01 and y1 <= c1 + 0.01)
        check["texts"].append(dict(kind=kind, fontsize_pt=float(art.get_fontsize()), text=art.get_text(),
                                   extent_in=[x0, y0, x1, y1], cell_bounds_in=bounds, font_ok=ok_font,
                                   within_cell=within))
        check["all_fonts_ok"] = bool(check["all_fonts_ok"] and ok_font)
        check["all_texts_within_cells"] = bool(check["all_texts_within_cells"] and within)
        boxes.append((kind, x0, y0, x1, y1))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                check["overlaps"].append([a[0], b[0], list(a[1:]), list(b[1:])])
    arr = np.asarray(Image.open(png).convert("RGB"))
    Hpx = arr.shape[0]
    sx = B.DPI / fdpi
    for b in bars:
        p0 = b["ax"].transData.transform((b["bx0"], b["by0"]))
        p1 = b["ax"].transData.transform((b["bx1"], b["by0"]))
        px0, px1, py = p0[0] * sx, p1[0] * sx, Hpx - p0[1] * sx
        predicted = float(px1 - px0)
        xs = slice(max(0, int(px0) - 40), min(arr.shape[1], int(px1) + 40))
        best = 0
        for dy in range(-6, 7):
            yy = int(round(py)) + dy
            if 0 <= yy < Hpx:
                dark = arr[yy, xs].max(axis=1) < 80
                run = cur = 0
                for v in dark:
                    cur = cur + 1 if v else 0
                    run = max(run, cur)
                best = max(best, run)
        check["scale_bars"].append(dict(row=b["row"], col=b["col"], slide=b["slide"], kind=b["kind"],
                                        bar_mm_on_page=round(b["bar_mm"], 3), predicted_px=predicted,
                                        measured_dark_run_px=int(best),
                                        measured_over_predicted=float(best / predicted)))
    check["scale_bar_tolerance"] = list(BAR_TOL)
    check["scale_bars_ok"] = bool(all(BAR_TOL[0] <= s["measured_over_predicted"] <= BAR_TOL[1] for s in check["scale_bars"]))
    check["pass"] = bool(check["all_fonts_ok"] and check["all_texts_within_cells"] and not check["overlaps"]
                         and check["scale_bars_ok"])
    plt.close(fig)
    (d / "legibility_check.json").write_text(json.dumps(check, indent=1))
    man = d / "panel_manifest.csv"
    pd.DataFrame(manifest).to_csv(man, index=False)
    summ = d / "matched_cost_summary.json"
    summ.write_text(json.dumps(summary, indent=1))

    note = ("CANDIDATE (not artwork of record): accepted Fig. 5 layout plus a matched-cost map column for BOTH slides "
            "of each pair (author decision 2026-11-13, option 2; 19:42 CST 'for all A B comparison'; R3.5). Map = "
            "mass-weighted mean cosine ground cost transported through each mosaic tile (column-wise on the neighbour, "
            "row-wise on the query) under the SAME saved "
            "exact OT plan that produced the displayed communities (plan sha256 = recipe plan_sha256; stored "
            "distance = recipe distance; cosine cost, uniform marginals and full support re-checked), projected to "
            "512-px patches via the bound mosaic metadata (one representative per patch), min-max normalised jointly "
            "over the two slides of the pair and inverted, painted jet/alpha 0.55 as the Fig. 6 ROI heatmap. "
            "Red = lowest matched cost. Not a tumour segmentation. H&E renders and community overlays are the "
            "SHA-256-authenticated recipe images; rank/distance line unchanged (item 4 parked). "
            f"Legibility check pass = {check['pass']}.")
    if absolute_cost:
        note = (
            "CANDIDATE, 2026-11-14 author-requested Figure 5 revision. Same layout, H&E, "
            "correspondence overlays, identities, ranks and distances as the 2026-11-13 "
            "both-sides candidate. Headers now say correspondence clusters. Matched-cost "
            "overlays use the retained patch costs, without solving OT or recomputing "
            "features or scientific endpoints. All six maps use the same numeric cosine-cost "
            "scale 0 to 1, with no clipping or per-pair normalization; red is lower cost, "
            "blue higher. Opacity remains 0.55 over H&E. Original plan-fidelity checks in "
            "the summary are inherited, not rerun. Source paths, hashes and patch-cost "
            "readbacks are recorded in display_readback.json. Neither display validates "
            "tissue classes or anatomical correspondence. The parked OT-distance labels "
            f"are unchanged. Legibility check pass = {check['pass']}."
        )
    write_provenance(d, [png, pdf, man, summ, d / "legibility_check.json"]
                     + [Path(sd["overlay_png"]) for v in summary.values() for sd in v["sides"].values()]
                     + [Path(sd["patch_values_csv"]) for v in summary.values() for sd in v["sides"].values()],
                     inputs, note, Path(sys.argv[0]) if absolute_cost else Path(__file__),
                     title=f"{figure} / {stem}")
    print(json.dumps(dict(png=str(png), sha256=sha256(png)[:16], size_px=list(Image.open(png).size),
                          page_mm=[B.WIDTH_MM, round(fig_h * 25.4, 1)], legibility_pass=check["pass"],
                          overlaps=len(check["overlaps"]), bars_ok=check["scale_bars_ok"],
                          fidelity={k: v["fidelity"]["pass"] for k, v in summary.items()},
                          cost_range={f"{k}/{r}": [round(sd["tile_cost_min"], 3), round(sd["tile_cost_median"], 3),
                                                   round(sd["tile_cost_max"], 3)]
                                      for k, v in summary.items() for r, sd in v["sides"].items()}), indent=1))
    if not check["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
