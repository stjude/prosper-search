"""Re-draw the three R3.5-L2 source panels used by Supplementary Fig. S-R3.5-L2
with publication labels (2026-11-04 supplementary-figure audit). Plotting only.

Sources (read-only, accepted run `$PROSPER_OUT/r3_5_localisation_maps_20261017/l2/run_003/`):
  overlay_D2_4288.png, overlay_D2_4298.png  <- per_tile_cost_D2.csv.gz (cost_i per reference OT row and
      P2 query), per_pair_D2.csv.gz (pair arm; duplicates kept, as in the run), per_reference_D2.csv.gz
      (group, discriminative AUROC/AP, row counts), the read-only WSI thumbnail and the L1 MapSlide
      projection (same code path as the run: scripts/r3_5_l1_maps.py + prosper localization_cli).
  matrix_D1_bin_same_group.png               <- matrices_D1.csv.
The two drawing functions are ports of `scripts/r3_5_l2_alignment.py::fig_overlay` / `fig_matrices`
with the label text parameterised; Delta_i is recomputed from the saved per-tile costs with the run's
own `discriminative_values` (mean other-group cost minus mean same-group cost), i.e. the quantity the
run plotted, not a new analysis.

Label changes (TEXT_REVISED vs TEXT_ORIGINAL): "tumour" -> "tumor", "nontumour" -> "non-tumor",
"artefact" -> "artifact", "nontumour_pooled" -> "non-tumor (pooled)"; overlay class legend limited to
classes with n > 0; matrix colour-bar label wrapped so it is not clipped. Suptitles (cropped off by the
assembly script) are unchanged.

Decision D1 (2026-11-04; TEXT_D1): the overlay's first sub-panel title shows the de-identified
slide ID ("reference <slide ID> (NB)") instead of the numeric scan key ("reference 4288 (NB)");
mapping _common.SCAN_TO_SLIDE_ID. The suptitle (cropped off by the assembly) and the file names keep the
scan key.

  --check-port DIR   re-draw with the original text into DIR and compare with the run_003 PNGs
                     (must be pixel-identical); exit 1 otherwise.
  --pre-d1 DIR       re-draw with the 2026-11-04 (pre-D1) text into DIR and compare with
                     sources_20261104/ (must be pixel-identical); exit 1 otherwise.
  (default)          write the D1 panels + SHA256SUMS.txt + PROVENANCE.md to
                     $PROSPER_OUT/figures_20261021/figS_r35_L2_discriminative_map/sources_20261104b/
                     (sources_20261104/ is left in place).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import out_dir, sha256, slide_id, write_provenance  # noqa: E402

REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("r3_5_l2_alignment", REPO / "scripts" / "r3_5_l2_alignment.py")
L2M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L2M)
L1 = L2M.L1
RUN = L2M.STAGE / "run_003"
REFS = ("4288", "4298")
MATRIX = ("bin", "same_group")

TEXT_ORIGINAL = dict(
    cname5=dict(L2M.CNAME5), cname2=dict(L2M.CNAME2),
    disc_title="discriminative map: tumour AUROC {auc:.3f}, AP {ap:.3f}",
    cbar_matrix="share of the row's transported mass", legend_n0=True, ref_title="reference {scan} ({group})")
TEXT_REVISED = dict(
    cname5={1: "tumor", 2: "immune", 3: "necrosis", 4: "non-tumor", 5: "artifact"},
    cname2={1: "tumor", 2: "non-tumor (pooled)"},
    disc_title="discriminative map: tumor AUROC {auc:.3f}, AP {ap:.3f}",
    cbar_matrix="share of the row's\ntransported mass", legend_n0=False, ref_title="reference {scan} ({group})")
TEXT_D1 = dict(TEXT_REVISED, ref_title="reference {sid} ({group})")


def load_deltas():
    cost = pd.read_csv(RUN / "per_tile_cost_D2.csv.gz", dtype={"reference": str, "query": str},
                       float_precision="round_trip")
    pairs = pd.read_csv(RUN / "per_pair_D2.csv.gz", dtype={"reference": str, "query": str})
    rr = pd.read_csv(RUN / "per_reference_D2.csv.gz", dtype={"reference": str})
    out = {}
    for r in REFS:
        c = cost[cost.reference == r]
        cost_by_q = {q: g.sort_values("ref_row").cost.to_numpy() for q, g in c.groupby("query")}
        p = pairs[pairs.reference == r]
        same = p[p.arm == "same_group"]["query"]; other = p[p.arm == "other_group"]["query"]
        row = rr[rr.reference == r].iloc[0].to_dict()
        assert len(same) == row["n_same_rows"] and len(other) == row["n_other_rows"], r
        delta = L2M.discriminative_values([cost_by_q[q] for q in same], [cost_by_q[q] for q in other])
        out[r] = (delta, row)
    return out


def fig_overlay(r, delta, rr, LC, meta, path: Path, text: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch, Rectangle
    import openslide
    CN = text["cname5"]
    ref = L1.MapSlide(r, meta[r], LC=LC)
    assert len(delta) == len(ref.row_label), (r, len(delta), len(ref.row_label))
    pv = L1.project(delta[ref.rows], ref.owners, "mean")
    ws = openslide.OpenSlide(str(L1.WSI).format(r))
    W0, H0 = ws.dimensions
    thumb = ws.get_thumbnail((1400, 1400))
    sc = thumb.size[0] / W0
    locs = np.array(ref.slide.patches, dtype=float)
    vmax = float(np.abs(pv).max()) or 1.0
    div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ec9a8a", "#b8322a"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), constrained_layout=True)
    for ax in axes:
        ax.imshow(thumb, alpha=1.0 if ax is axes[0] else 0.35)
        ax.set_axis_off()
    axes[0].set_title(text["ref_title"].format(scan=r, sid=slide_id(r), group=rr.get("group", "")), fontsize=9)
    rects = [Rectangle((x * sc, y * sc), 512 * sc, 512 * sc) for x, y in locs]
    pc = PatchCollection(rects, cmap=div, norm=Normalize(-vmax, vmax), linewidth=0)
    pc.set_array(pv)
    axes[1].add_collection(pc)
    fig.colorbar(pc, ax=axes[1], shrink=0.7, label="Delta = mean other-group cost - mean same-group cost")
    axes[1].set_title(text["disc_title"].format(auc=rr.get("disc_auroc_A", float("nan")),
                                                ap=rr.get("disc_ap_A", float("nan"))), fontsize=9)
    lab = ref.patch_label
    for c, col in L2M.CLASS_COLOURS.items():
        sel = lab == c
        if sel.any():
            axes[2].add_collection(PatchCollection([Rectangle((x * sc, y * sc), 512 * sc, 512 * sc) for x, y in locs[sel]],
                                                   facecolor=col, edgecolor="none", alpha=0.85))
    axes[2].legend(handles=[Patch(color=L2M.CLASS_COLOURS[c], label=f"{CN[c]} (n={int((lab == c).sum())})") for c in CN
                            if text["legend_n0"] or int((lab == c).sum()) > 0],
                   loc="lower left", fontsize=7, frameon=True)
    axes[2].set_title("pathologist annotation classes (unannotated patches not drawn)", fontsize=9)
    fig.suptitle(f"R3.5-L2 design 2, reference {r}. Rule: per reference tile, cost_i = transport-weighted mean cosine cost "
                 f"of its exact-OT links (full plan); Delta_i = mean of cost_i over the reference's {rr['n_other_rows']} "
                 f"other-group P2 queries minus mean over its {rr['n_same_rows']} same-group P2 queries (positive, red = matched "
                 "better by same-diagnosis slides); patch value = mean of its owning mosaic tiles. Reference choice: the "
                 "Fig. 5 reference slides are not P2 references, so the two P2 references of lowest numeric id are shown. "
                 "No manual choice.", fontsize=8, wrap=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_matrix(mat_df: pd.DataFrame, gran: str, arm: str, path: Path, text: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", L2M.SEQ)
    cmap.set_bad("#ffffff")
    data_names = L2M.CNAME5 if gran == "5" else L2M.CNAME2          # keys in matrices_D1.csv
    shown = text["cname5"] if gran == "5" else text["cname2"]       # display text
    keys = sorted(data_names)
    labs = [data_names[c] for c in keys]
    disp = [shown[c] for c in keys]
    x = mat_df[(mat_df.granularity == gran) & (mat_df.arm == arm)]
    fig, axes = plt.subplots(1, 3, figsize=(3.6 * 3 + 1, 3.9 if gran == "5" else 3.0), constrained_layout=True)
    for ax, kind in zip(axes, ("observed", "independence", "ceiling")):
        xx = x[x.kind == kind]
        A = np.full((len(labs), len(labs)), np.nan)
        N = np.zeros(len(labs), int)
        for a_, ra in enumerate(labs):
            for b_, cb in enumerate(labs):
                v = xx[(xx.ref_class == ra) & (xx.query_class == cb)]
                if len(v):
                    A[a_, b_] = v.value.iloc[0]
                    N[a_] = int(v.n_pairs.iloc[0])
        im = ax.imshow(np.ma.masked_invalid(A), vmin=0, vmax=1, cmap=cmap)
        for a_ in range(len(labs)):
            for b_ in range(len(labs)):
                if np.isfinite(A[a_, b_]):
                    ax.text(b_, a_, f"{A[a_, b_]:.2f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if A[a_, b_] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(disp, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels([f"{l} (n={n})" for l, n in zip(disp, N)], fontsize=7)
        ax.set_xlabel("query-tile class", fontsize=8)
        ax.set_title(kind, fontsize=9)
    axes[0].set_ylabel("reference-tile class", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label=text["cbar_matrix"])
    fig.suptitle(f"R3.5-L2 design 1, {'5-class' if gran == '5' else 'binary'}, P2 {arm.replace('_', '-')} pairs: "
                 "row-normalised class-to-class transport (full exact plan, links with both tiles annotated), "
                 "mean over pairs; n = pairs with that reference class present. Ceiling = min(p,q)/p on the "
                 "diagonal (off-diagonal not fixed by the marginals).", fontsize=7, wrap=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_all(d: Path, text: dict) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    LC, _ = L1.import_prosper()
    meta = L1.meta_cache_load(L2M.BIND / "meta_tiles.pkl")
    written = []
    for r, (delta, rr) in load_deltas().items():
        p = d / f"overlay_D2_{r}.png"
        fig_overlay(r, delta, rr, LC, meta, p, text)
        written.append(p)
    mat = pd.read_csv(RUN / "matrices_D1.csv")
    p = d / f"matrix_D1_{MATRIX[0]}_{MATRIX[1]}.png"
    fig_matrix(mat, MATRIX[0], MATRIX[1], p, text)
    written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-port", metavar="DIR")
    ap.add_argument("--pre-d1", metavar="DIR")
    a = ap.parse_args()
    if a.check_port or a.pre_d1:
        from PIL import Image, ImageChops
        ok = True
        ref_dir = RUN if a.check_port else out_dir("figS_r35_L2_discriminative_map") / "sources_20261104"
        for p in render_all(Path(a.check_port or a.pre_d1), TEXT_ORIGINAL if a.check_port else TEXT_REVISED):
            x = Image.open(ref_dir / p.name).convert("RGBA"); y = Image.open(p).convert("RGBA")
            same = x.size == y.size and ImageChops.difference(x, y).getbbox() is None
            bbox = None if x.size != y.size else ImageChops.difference(x, y).getbbox()
            print(p.name, "pixel-identical" if same else f"DIFFERENT {x.size} {y.size} bbox={bbox}",
                  "| byte-identical" if sha256(p) == sha256(ref_dir / p.name) else "| bytes differ")
            ok &= same
        sys.exit(0 if ok else 1)
    d = out_dir("figS_r35_L2_discriminative_map") / "sources_20261104b"
    written = render_all(d, TEXT_D1)
    inputs = [RUN / n for n in ("per_tile_cost_D2.csv.gz", "per_pair_D2.csv.gz", "per_reference_D2.csv.gz",
                                "matrices_D1.csv", "COMPLETE.json", "overlay_D2_4288.png", "overlay_D2_4298.png",
                                "matrix_D1_bin_same_group.png")] + [L2M.BIND / "meta_tiles.pkl"]
    write_provenance(d, written, inputs,
                     "Plotting-only re-draw of the three l2/run_003 panels with reviewer-facing labels (tumor, non-tumor, "
                     "artifact, non-tumor (pooled); class legend limited to n > 0; matrix colour-bar label wrapped). Ports "
                     "of r3_5_l2_alignment.fig_overlay / fig_matrices; the original-text re-draw (--check-port) is compared "
                     "pixel-wise with the run_003 PNGs. Delta_i from the saved per-tile costs with the run's "
                     "discriminative_values. Decision D1 (2026-11-04): the first sub-panel title shows the slide ID "
                     f"({slide_id(4288)} / {slide_id(4298)}) instead of the scan key 4288 / 4298; otherwise identical to "
                     "sources_20261104/.", Path(__file__), title="figS_r35_L2 sources, D1 slide IDs (2026-11-04)")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
