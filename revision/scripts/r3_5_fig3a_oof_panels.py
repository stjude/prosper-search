#!/usr/bin/env python3
"""R2.4 / R3.5 Fig. 3A replacement panels: pathologist tile annotation beside OUT-OF-FOLD tumour probability.

Design decision Q5(a) (thin_reset_plan §13.4): 12151 (EWS; main panel) and 7968 (WT; supplementary failure panel),
drawn from the pooled 5-fold slide-GroupKFold out-of-fold probabilities of job 323489975 (`s16_grouped/oof_probs.npy`)
on ANNOTATED TILES ONLY; unannotated tissue is left uncoloured. No model is loaded or run.

Per scan, three panels at WSI level 2 (actual OpenSlide downsample): H&E; annotation grid (1 Tumor green, 4 Non-tumor
blue, 3 Necrosis orange, 2 Immune purple, 5 Artifacts grey); OOF P(tumour) (blue 0 -> red 1) with a thin contour of
the argmax decision (pred == Tumor). Legend metrics recomputed from the saved arrays: n tiles, class counts, OOF AUROC,
sensitivity/specificity at argmax and at P>=0.5, necrosis-called-tumour fraction; plus the across-slide context
(per-slide OOF AUROC over slides with both classes annotated; rank among slides with >=200 tiles of each class).

Inputs are hashed into metrics.json. Label cross-check against comet-annotation/output/pathologist_annotated_labels/<scan>.csv
is asserted (every OOF tile matches a nonzero-label row at the same level-0 coordinate).
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = "/PROSPER_WORKSPACE"
A = f"{H}/comet-annotation"
SPLIT = f"{A}/output/pathologist_annotated_train_test_split_uni2"
LABELS = f"{A}/output/pathologist_annotated_labels"
WSI_DIR = f"{H}/data/comet_final_HPC"
SCANS = {"12151": "main_EWS", "7968": "supp_failure_WT"}
TILE = 512
LEVEL = 2
NAMES = {1: "Tumor", 2: "Immune cells", 3: "Necrosis", 4: "Non-tumor", 5: "Artifacts"}
ANN_RGB = {1: (0, 158, 115), 4: (0, 114, 178), 3: (230, 159, 0), 2: (128, 0, 128), 5: (140, 140, 140)}
EXPECTED_TUMOR_OOF_AUROC = 0.9457737203096972


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_oof(out_root):
    inputs = {
        "oof_probs": f"{out_root}/s16_grouped/oof_probs.npy",
        "y_train_rand": f"{SPLIT}/y_train_rand.npy", "y_test_rand": f"{SPLIT}/y_test_rand.npy",
        "idx_train_rand": f"{SPLIT}/idx_train_rand.npy", "idx_test_rand": f"{SPLIT}/idx_test_rand.npy",
        "slide_info": f"{SPLIT}/slide_info.csv",
    }
    hashes = {k: sha256_file(v) for k, v in inputs.items()}
    oof = np.load(inputs["oof_probs"])
    y = np.concatenate([np.load(inputs["y_train_rand"]), np.load(inputs["y_test_rand"])])
    idx = np.concatenate([np.load(inputs["idx_train_rand"]), np.load(inputs["idx_test_rand"])])
    info = pd.read_csv(inputs["slide_info"], dtype={"slide_id": str})
    if sorted(idx.tolist()) != list(range(len(info))):
        raise RuntimeError("concat(idx_train_rand, idx_test_rand) is not a permutation of slide_info rows")
    info = info.iloc[idx].reset_index(drop=True)
    if oof.shape != (len(y), 5) or len(info) != len(y):
        raise RuntimeError("OOF / label / coordinate row counts disagree")
    classes = np.unique(y)
    if list(classes) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"unexpected class codes {classes}")
    auc = float(roc_auc_score((y == 1).astype(int), oof[:, 0]))
    if abs(auc - EXPECTED_TUMOR_OOF_AUROC) > 1e-9:
        raise RuntimeError(f"pooled tumour OOF AUROC {auc} != expected")
    frame = info.assign(label=y, p_tumour=oof[:, 0], pred=classes[oof.argmax(1)])
    return frame, hashes, inputs


def slide_metrics(f):
    t = f["label"].to_numpy() == 1
    both = t.any() and (~t).any()
    m = dict(n_tiles=int(len(f)), n_tumour=int(t.sum()), n_non_tumour=int((~t).sum()),
             class_counts={NAMES[c]: int((f["label"] == c).sum()) for c in NAMES},
             oof_auroc=float(roc_auc_score(t.astype(int), f["p_tumour"])) if both else None)
    for rule, tp in (("argmax", f["pred"].to_numpy() == 1), ("p_ge_0.5", f["p_tumour"].to_numpy() >= 0.5)):
        m[f"sensitivity_{rule}"] = float((tp & t).sum() / t.sum()) if t.any() else None
        m[f"specificity_{rule}"] = float((~tp & ~t).sum() / (~t).sum()) if (~t).any() else None
    nec = f["label"].to_numpy() == 3
    m["n_necrosis"] = int(nec.sum())
    m["necrosis_called_tumour_argmax"] = float(((f["pred"].to_numpy() == 1) & nec).sum() / nec.sum()) if nec.any() else None
    return m


def context(frame):
    rows = []
    for s, g in frame.groupby("slide_id"):
        t = g["label"].to_numpy() == 1
        nec = g["label"].to_numpy() == 3
        if t.any() and (~t).any():
            rows.append(dict(slide=s, n_tumour=int(t.sum()), n_non=int((~t).sum()), n_necrosis=int(nec.sum()),
                             auroc=float(roc_auc_score(t.astype(int), g["p_tumour"])),
                             necrosis_called_tumour=float(((g["pred"].to_numpy() == 1) & nec).sum() / nec.sum()) if nec.any() else np.nan))
    d = pd.DataFrame(rows)
    nec_rows = []
    for s, g in frame.groupby("slide_id"):
        nec = g["label"].to_numpy() == 3
        if nec.sum() >= 200:
            nec_rows.append(dict(slide=s, n_necrosis=int(nec.sum()),
                                 necrosis_called_tumour=float(((g["pred"].to_numpy() == 1) & nec).sum() / nec.sum())))
    necd = pd.DataFrame(nec_rows).sort_values("necrosis_called_tumour", ascending=False).reset_index(drop=True)
    pooled_nec = frame["label"].to_numpy() == 3
    pooled_nec_rate = float(((frame["pred"].to_numpy() == 1) & pooled_nec).sum() / pooled_nec.sum())
    dall = d.sort_values("auroc", ascending=False).reset_index(drop=True)
    big = d[(d.n_tumour >= 200) & (d.n_non >= 200)].sort_values("auroc", ascending=False).reset_index(drop=True)
    ctx = dict(n_slides_both_classes=int(len(d)), median=float(d.auroc.median()),
               q1=float(d.auroc.quantile(0.25)), q3=float(d.auroc.quantile(0.75)), min=float(d.auroc.min()),
               n_slides_ge200_each=int(len(big)), median_ge200=float(big.auroc.median()),
               rank_ge200={s: int(big.index[big.slide == s][0]) + 1 for s in SCANS if (big.slide == s).any()},
               rank_all={s: int(dall.index[dall.slide == s][0]) + 1 for s in SCANS if (dall.slide == s).any()},
               pooled_necrosis_called_tumour=pooled_nec_rate, n_slides_ge200_necrosis=int(len(necd)),
               necrosis_rank_ge200={s: int(necd.index[necd.slide == s][0]) + 1 for s in SCANS if (necd.slide == s).any()},
               necrosis_median_ge200=float(necd.necrosis_called_tumour.median()))
    return ctx, d, necd


def strip_plot(d, necd, out_png, ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(20261008)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, df, col, ylab, ttl in ((axes[0], d, "auroc", "per-slide held-out AUROC (tumour vs rest)", f"{len(d)} slides with both classes"),
                                   (axes[1], necd, "necrosis_called_tumour", "fraction of necrosis tiles called tumour", f"{len(necd)} slides with >=200 necrosis tiles")):
        df = df.reset_index(drop=True)
        x = rng.uniform(-0.12, 0.12, len(df))
        ax.scatter(x, df[col], s=12, color="grey", alpha=0.7, linewidths=0)
        ax.hlines(df[col].median(), -0.25, 0.25, color="black", linewidth=1)
        for s, mk, lab in (("12151", "D", "12151 (main panel)"), ("7968", "s", "7968 (failure panel)")):
            hit = df.index[df.slide == s]
            if len(hit):
                ax.scatter([x[hit[0]]], [df.loc[hit[0], col]], s=48, marker=mk, facecolor="none", edgecolor="red", linewidths=1.2, label=lab)
        ax.set_xlim(-0.5, 0.5); ax.set_xticks([]); ax.set_ylabel(ylab, fontsize=8); ax.set_title(ttl, fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc="lower left" if col == "auroc" else "upper left", frameon=False)
    axes[1].axhline(ctx["pooled_necrosis_called_tumour"], color="grey", linestyle=":", linewidth=1)
    axes[1].text(0.27, ctx["pooled_necrosis_called_tumour"], f"pooled {ctx['pooled_necrosis_called_tumour']:.2f}", fontsize=7, va="center")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def check_labels(scan, f):
    path = f"{LABELS}/{scan}.csv"
    lab = pd.read_csv(path, dtype={"slide_id": str})
    nz = lab[lab.patch_label != 0]
    key = dict(zip(zip(nz.wsi_loc_x.astype(int), nz.wsi_loc_y.astype(int)), nz.patch_label.astype(int)))
    mism = sum(1 for x, yy, l in zip(f.wsi_loc_x.astype(int), f.wsi_loc_y.astype(int), f.label) if key.get((x, yy)) != int(l))
    if mism or len(nz) != len(f):
        raise RuntimeError(f"{scan}: label CSV cross-check failed (mismatches {mism}, nonzero rows {len(nz)} vs OOF {len(f)})")
    return dict(path=path, sha256=sha256_file(path), rows=int(len(lab)), nonzero_rows=int(len(nz)),
                n_unannotated_grid_tiles=int((lab.patch_label == 0).sum()))


def render(scan, f, out_png, metrics):
    import openslide
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors, cm
    from PIL import Image

    path = f"{WSI_DIR}/{scan}.svs"
    sl = openslide.OpenSlide(path)
    ds = float(sl.level_downsamples[LEVEL])
    W, Hh = sl.level_dimensions[LEVEL]
    he = sl.read_region((0, 0), LEVEL, (W, Hh)).convert("RGB")
    he_np = np.asarray(he)
    # bounding box of annotated tiles with margin, in level-L px
    xs, ys = f.wsi_loc_x.to_numpy() / ds, f.wsi_loc_y.to_numpy() / ds
    tl = TILE / ds
    x0, x1 = max(0, int(xs.min() - 4 * tl)), min(W, int(xs.max() + 5 * tl))
    y0, y1 = max(0, int(ys.min() - 4 * tl)), min(Hh, int(ys.max() + 5 * tl))
    crop = he_np[y0:y1, x0:x1]

    ann = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    heat = np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for x, y, lab, p, pr in zip(xs, ys, f.label, f.p_tumour, f.pred):
        a, b = int(round(x)) - x0, int(round(y)) - y0
        c, d = a + int(round(tl)), b + int(round(tl))
        r, g, bb = ANN_RGB[int(lab)]
        ann[b:d, a:c] = (r, g, bb, 170)
        heat[b:d, a:c] = p
        mask[b:d, a:c] = 1 if pr == 1 else 0
    cmap = matplotlib.colormaps["coolwarm"]
    heat_rgba = cmap(np.nan_to_num(heat, nan=0.0))
    heat_rgba[..., 3] = np.where(np.isnan(heat), 0.0, 0.85)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    for ax in axes:
        ax.imshow(crop); ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title(f"H&E (scan {scan}, downsample {ds:.0f})")
    axes[1].imshow(ann); axes[1].set_title("Pathologist tile annotation")
    axes[2].imshow(heat_rgba); axes[2].set_title("Cross-validated probability of tumour class (annotated tiles only)")
    annotated = ~np.isnan(heat)
    axes[2].contour(np.ma.masked_where(~annotated, mask.astype(float)), levels=[0.5], colors="black", linewidths=0.4)
    # scale bar (2 mm) from level-0 mpp x actual downsample
    mpp = float(sl.properties.get("openslide.mpp-x")) * ds
    bar_px = 2000.0 / mpp
    for ax in axes:
        ax.plot([20, 20 + bar_px], [crop.shape[0] - 25] * 2, color="black", linewidth=3)
        ax.text(20, crop.shape[0] - 40, "2 mm", fontsize=8)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=np.array(ANN_RGB[k]) / 255, label=NAMES[k]) for k in (1, 4, 3, 2, 5)
                            if metrics["class_counts"][NAMES[k]] > 0] + [Patch(facecolor="white", edgecolor="grey", label="unannotated (no prediction)")],
                   loc="lower right", fontsize=8)
    sm = cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[2], fraction=0.03, pad=0.01); cb.set_label("probability of tumour class (held-out fold)")
    auc = metrics["oof_auroc"]
    comp = ", ".join(f"{v:,} {k.lower()}" for k, v in metrics["class_counts"].items() if v > 0)
    axes[2].text(0.01, 0.01, f"n={metrics['n_tiles']:,} tiles ({comp}); "
                 f"held-out AUROC {auc:.3f}; sens {metrics['sensitivity_argmax']:.3f} / spec {metrics['specificity_argmax']:.3f} (most-probable class); "
                 "black line = boundary of tumour calls within annotated tiles", transform=axes[2].transAxes, fontsize=7,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return dict(wsi=path, wsi_sha256=sha256_file(path), level=LEVEL, actual_downsample=ds, level_dims=[W, Hh],
                crop_level_px=[x0, y0, x1, y1], mpp_x=sl.properties.get("openslide.mpp-x"),
                vendor=sl.properties.get("openslide.vendor"))


def main():
    t0 = time.time()
    out_root = os.environ.get("PROSPER_OUT")
    if not out_root:
        raise RuntimeError("PROSPER_OUT must be exported")
    OUT = os.path.join(out_root, "r3_5_fig3a_oof_panels_20261008_v2")
    os.makedirs(OUT, exist_ok=False)
    frame, hashes, inputs = load_oof(out_root)
    ctx, per_slide, necd = context(frame)
    fold_path = f"{out_root}/s16_grouped/per_slide_oof.csv"
    folds = pd.read_csv(fold_path, dtype={"slide": str})[["slide", "fold"]]
    hashes["per_slide_oof_fold"] = sha256_file(fold_path)
    inputs["per_slide_oof_fold"] = fold_path
    per_slide = per_slide.merge(folds, on="slide", how="left")
    if per_slide.fold.isna().any():
        raise RuntimeError("fold missing for some slides")
    per_slide.sort_values("auroc", ascending=False).to_csv(f"{OUT}/per_slide_oof_auroc_both_classes.csv", index=False)
    necd.merge(folds, on="slide", how="left").to_csv(f"{OUT}/per_slide_necrosis_called_tumour_ge200.csv", index=False)
    strip_plot(per_slide, necd, f"{OUT}/fig3a_supp_per_slide_strip.png", ctx)
    rec = dict(script=os.path.basename(__file__), script_sha256=sha256_file(os.path.abspath(__file__)),
               input_sha256=hashes, inputs=inputs, pooled_tumour_oof_auroc=EXPECTED_TUMOR_OOF_AUROC,
               library_versions={m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "matplotlib", "openslide", "PIL")},
               python=sys.version,
               across_slide_context=ctx, scans={})
    for scan, role in SCANS.items():
        f = frame[frame.slide_id == scan].reset_index(drop=True)
        if f.empty:
            raise RuntimeError(f"no OOF rows for {scan}")
        m = slide_metrics(f)
        m["label_csv_check"] = check_labels(scan, f)
        f.to_csv(f"{OUT}/tiles_{scan}.csv", index=False)
        m["render"] = render(scan, f, f"{OUT}/fig3a_{role}_{scan}.png", m)
        m["role"] = role
        rec["scans"][scan] = m
        print(scan, role, {k: m[k] for k in ("n_tiles", "n_tumour", "n_non_tumour", "oof_auroc", "sensitivity_argmax", "specificity_argmax", "necrosis_called_tumour_argmax")}, flush=True)
    rec["runtime_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(f"{OUT}/metrics.json", "w"), indent=1)
    with open(f"{OUT}/README.md", "w") as fh:
        fh.write("# Fig. 3A OOF panels (R2.4 / R3.5)\n\n" + __doc__ + f"\nAcross-slide context: {json.dumps(ctx)}\nRuntime {rec['runtime_s']} s.\n")
    with open(f"{OUT}/sha256.txt", "w") as fh:
        for n in sorted(os.listdir(OUT)):
            if n not in ("sha256.txt", "COMPLETE"):
                fh.write(f"{sha256_file(os.path.join(OUT, n))}  {n}\n")
    open(f"{OUT}/COMPLETE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print("wrote", OUT, f"{rec['runtime_s']} s")


if __name__ == "__main__":
    sys.exit(main())
