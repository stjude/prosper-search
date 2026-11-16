"""Fig. 3 (revised): tumor-only versus full-slide embeddings for slide search.

[internal record] block R2.1-T6-A as amended on 2026-10-21 (≈13:40:
A review comment asked that Fig. 3 "include ground-truth annotation by pathologist
for comparison", so the tissue example stays in the main figure; ≈14:08:
publication-grade, no internal wording, St. Jude identifier, classifier
quality / 5 %-budget panels moved to Supplementary Fig. S-R2.1-F4 because no
review comment asked for them and they duplicate Table S16 / Table S-R2.1-B;
≈14:48: the per-tumor-group panel of the submitted Fig. 3B returns as panel
C, now with standard metrics, patient-held-out masks and bootstrap
intervals — the three defects that had led the 10-17 spec to drop it).

Panel A  one library slide (Ewing sarcoma; slide and patient IDs from the
         local figure-ID file, patient fold 3): H&E at WSI level 2; the
         pathologist's tile annotation; the probability of the tumor class
         from the TF-2 UNI2 100 %-budget head fitted WITHOUT this patient
         (fold-3 head, held-out rows); the boundary of the
         most-probable-class tumor call is no longer drawn (revision,
         2026-11-10). Only annotated tiles carry a
         prediction. The same held-out head family produces the masks
         evaluated in panel B.
Panel B  filtered minus all-tissue change in macro hit@5 (left) and
         precision@5 (right), 100 % budget, tumor-only query AND library
         (the pre-registered headline arm; author ≈15:30 dropped the
         query-only arm from the main figure as distracting), 7 settings
         (encoder x method), whole-patient bootstrap 95 % intervals.
Panel C  (design decision 2026-11-04, revision: candidate
         `forest_rer` of fig3_panelC_candidates.py adopted) forest plot of the
         relative error reduction (RER) in top-5 plurality-vote F1 by
         diagnostic group: one column per tile encoder (UNI2, Virchow2,
         H-Optimus-1, TITAN), Optimal Transport and Yottixel side by side
         within a column; top row = all groups (macro, identical to B), then
         the 14 groups ordered by number of queries; point + 95 %
         whole-patient bootstrap interval; filled dot = interval excludes
         zero, open dot = includes zero, grey cross = RER interval not
         estimable (point only). Per-group hit@5 / precision@5 stay in
         Supplementary Table S-R2.1-D (source_data_panelC_groups_all_arms.csv).
         The 2026-10-21 heat-map C (hit@5 | precision@5 | RER) is kept behind
         `--panel_c=heatmap` (written to its own variant directory) and
         verbatim in superseded/fig3_tumor_filtering_heatmapC_20261021.py.
         `--layout=AB` writes the two-panel alternative to a separate
         directory; `check_fig3_panelC_groups.py` recomputes every group
         hit@5 / precision@5 point estimate from the per-query table.

All numbers (n tiles, AUROC, sensitivity, specificity, n queries, k, interval
type) belong to the legend, not the figure. The classifier-quality bars and
the query-only arm and the 5 %-budget panel are drawn by
`figS_r21_f4_classifier_budget.py`, which
imports the loaders and panel functions from this module.

Sources (accepted, read-only):
  s2/run_002/heads/uni2/b100/fold3/{held_out_rows.csv.gz,probabilities.npy,
      predictions.npy,training_tiles.csv.gz}                  (A)
  comet-annotation/output/pathologist_annotated_labels/12151.csv   (A: label check)
  data/comet_final_HPC/12151.svs                              (A: H&E)
  data/ALIdf3Ali.csv                                          (A: St. Jude ID)
  s6/evaluation_primary_001/contrasts.csv                     (B, C)
  s6/evaluation_primary_001/wording_classes.csv               (B cross-check)
  writing_001/fig3_panel_{B}_data.csv                         (cross-check only)
  writing_001/S_R2_1_D_headline_cell_groups.csv               (C: group n check)
  s2/run_002/classifier_report_001/*.csv, writing_001/fig3_panel_{A,C}_data.csv
                                                              (S-R2.1-F4 only)

No model is loaded or run. Panel A metrics are recomputed from the stored
held-out probabilities; every other plotted value is cross-checked against
the writer's panel-data extracts and the wording-class table. Encoders are
shown in table order (UNI2, Virchow2, H-Optimus-1, TITAN), not ranked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AGG_COLORS, FIGURE_IDS, GREY, prosper_out, save, sha256, slide_id, style  # noqa: E402

TF2 = prosper_out() / "r2_1_filter_tf2_20261016"

# Panel A example (design decision Q5(a), 2026-10-09; kept in the main figure 2026-10-21).
EXAMPLE_SLIDE, EXAMPLE_PATIENT, EXAMPLE_GROUP, EXAMPLE_FOLD = "12151", FIGURE_IDS["fig3_example_patient"], "EWS", 3
EXAMPLE_SJID, EXAMPLE_GROUP_NAME = slide_id(EXAMPLE_SLIDE), "Ewing sarcoma"  # manuscript convention: Comp.Bio.ID
METADATA = prosper_out().parent / "data/ALIdf3Ali.csv"  # COMET master metadata (untracked input)
HEAD_DIR = TF2 / f"s2/run_002/heads/uni2/b100/fold{EXAMPLE_FOLD}"
HELD_OUT_ROWS = HEAD_DIR / "held_out_rows.csv.gz"
HEAD_PROBS = HEAD_DIR / "probabilities.npy"
HEAD_PREDS = HEAD_DIR / "predictions.npy"
HEAD_TRAIN = HEAD_DIR / "training_tiles.csv.gz"
HAORAN = Path("/PROSPER_WORKSPACE")
LABEL_CSV = HAORAN / f"comet-annotation/output/pathologist_annotated_labels/{EXAMPLE_SLIDE}.csv"
WSI = HAORAN / f"data/comet_final_HPC/{EXAMPLE_SLIDE}.svs"
WSI_LEVEL, TILE_PX, TUMOUR_CLASS = 2, 512, 1
ANN_NAMES = {1: "Tumor", 2: "Immune cells", 3: "Necrosis", 4: "Non-tumor", 5: "Artifacts"}
ANN_RGB = {1: (0, 158, 115), 4: (0, 114, 178), 3: (230, 159, 0), 2: (128, 0, 128), 5: (140, 140, 140)}
TILE_METRICS = TF2 / "s2/run_002/classifier_report_001/tile_metrics.csv"
IN_SAMPLE = TF2 / "s2/run_002/classifier_report_001/s2b_full_noqc_vs_full.csv"
CONTRASTS = TF2 / "s6/evaluation_primary_001/contrasts.csv"
WORDING = TF2 / "s6/evaluation_primary_001/wording_classes.csv"
PANEL_DATA = {p: TF2 / f"writing_001/fig3_panel_{p}_data.csv" for p in "ABC"}
FIGURE = "fig3_tumor_filtering"
GROUP_SUPPORT = TF2 / "writing_001/S_R2_1_D_headline_cell_groups.csv"
RER_CONTRASTS = TF2 / "s6/evaluation_primary_001/rer_contrasts.csv"
RER_VMAX = 0.5  # colour limit for the RER heat map (max |point| 0.78, RB / H-Optimus-1 OT, clipped)
METRIC_LABEL = {"hit": "Hit@5", "precision": "Precision@5", "rer": "Relative error reduction\n(top-5 F1)"}
RER_REBASED = prosper_out() / "figures_20261021/fig3_rer_rebased_derivation/rer_rebased_contrasts.csv"  # derive_fig3_rer_precision.py
GROUP_VMAX = 0.12  # symmetric colour limit for panel C; max |point| over all cells is 0.121 (clipped cells are marked)

ENCODERS = [("uni2", "UNI2"), ("virchow2", "Virchow2"), ("h_optimus_1", "H-Optimus-1"), ("conch_v15", "TITAN\n(CONCH v1.5)")]  # display name per the Fig. 2 decision D3 (2026-11-03): TITAN first, its patch encoder in parentheses; constant updated 2026-11-04 (supplementary-figure audit follow-up); drawn only by panel_classifier (S-R2.1-F4a), not by the adopted Fig. 3
ENC_LABEL_BC = {"uni2": "UNI2", "virchow2": "Virchow2", "h_optimus_1": "H-Optimus-1", "titan": "TITAN"}
# Fixed order within an encoder block (table order, not performance order).
SETTINGS = [
    ("uni2", "ot_cosine", "query_only"), ("uni2", "ot_cosine", "both_filtered"),
    ("uni2", "yottixel", "query_only"), ("uni2", "yottixel", "both_filtered"),
    ("virchow2", "ot_cosine", "query_only"), ("virchow2", "ot_cosine", "both_filtered"),
    ("virchow2", "yottixel", "query_only"), ("virchow2", "yottixel", "both_filtered"),
    ("h_optimus_1", "ot_cosine", "query_only"), ("h_optimus_1", "ot_cosine", "both_filtered"),
    ("h_optimus_1", "yottixel", "query_only"), ("h_optimus_1", "yottixel", "both_filtered"),
    ("titan", "titan", "query_only"), ("titan", "titan", "both_filtered"),
]
# Main-figure arm (author, 2026-10-21 ≈15:30): the pre-registered headline arm, tumor-only query AND library.
# The query-only arm is drawn in Supplementary Fig. S-R2.1-F4 only.
MAIN_ARM = "both_filtered"
MAIN_SETTINGS = [s for s in SETTINGS if s[2] == MAIN_ARM]
ENC_BLOCKS = [("uni2", "UNI2"), ("virchow2", "Virchow2"), ("h_optimus_1", "H-Optimus-1"), ("titan", "TITAN")]
METHOD_COLOR = {"ot_cosine": AGG_COLORS["Optimal Transport"], "yottixel": AGG_COLORS["Yottixel"], "titan": AGG_COLORS["TITAN"]}
METHOD_LABEL = {"ot_cosine": "Optimal Transport", "yottixel": "Yottixel", "titan": "TITAN"}
SENS_COLOR, SPEC_COLOR = "#1f77b4", "#2ca02c"
TOL = 1e-9


# ----------------------------------------------------------------------------- loaders
def load_example() -> tuple[pd.DataFrame, dict]:
    """Held-out tiles of the example slide from the fold head that never saw its patient."""
    rows = pd.read_csv(HELD_OUT_ROWS, dtype={"slide_id": str})
    probs = np.load(HEAD_PROBS)
    preds = np.load(HEAD_PREDS)
    assert probs.shape == (len(rows), 5) and preds.shape == (len(rows),)
    assert (rows.position.to_numpy() == np.arange(len(rows))).all(), "held_out_rows.position is not the array row"
    assert (preds == probs.argmax(1) + 1).all(), "predictions are not argmax + 1"
    f = rows[rows.slide_id == EXAMPLE_SLIDE].copy()
    assert len(f) > 0 and (f.patient_id == EXAMPLE_PATIENT).all()
    f["p_tumour"] = probs[f.position.to_numpy(), TUMOUR_CLASS - 1]
    f["pred"] = preds[f.position.to_numpy()]
    f = f.rename(columns={"tissue_label": "label"})[["slide_id", "patient_id", "x", "y", "label", "p_tumour", "pred"]]
    f = f.sort_values(["y", "x"]).reset_index(drop=True)
    # The fold head must not have trained on this patient or slide.
    train = pd.read_csv(HEAD_TRAIN, dtype={"slide_id": str}, usecols=["slide_id", "patient_id"])
    assert not (train.patient_id == EXAMPLE_PATIENT).any() and not (train.slide_id == EXAMPLE_SLIDE).any()
    # Every held-out tile is a non-zero-label row of the pathologist's label CSV at the same level-0 coordinate.
    lab = pd.read_csv(LABEL_CSV)
    nz = lab[lab.patch_label != 0]
    key = dict(zip(zip(nz.wsi_loc_x.astype(int), nz.wsi_loc_y.astype(int)), nz.patch_label.astype(int)))
    assert len(nz) == len(f) and all(key.get((int(x), int(y))) == int(l) for x, y, l in zip(f.x, f.y, f.label))
    meta = pd.read_csv(METADATA, dtype=str, low_memory=False, usecols=["Slide.scan", "Comp.Bio.ID", "UID.Subject", "Tumor.group"])
    meta = meta[meta["Slide.scan"] == EXAMPLE_SLIDE]
    assert len(meta) == 1 and meta.iloc[0]["Comp.Bio.ID"] == EXAMPLE_SJID and meta.iloc[0]["UID.Subject"] == EXAMPLE_PATIENT \
        and meta.iloc[0]["Tumor.group"] == EXAMPLE_GROUP, meta
    t = f.label.to_numpy() == TUMOUR_CLASS
    call = f.pred.to_numpy() == TUMOUR_CLASS
    m = dict(slide=EXAMPLE_SLIDE, sj_id=EXAMPLE_SJID, patient=EXAMPLE_PATIENT, group=EXAMPLE_GROUP, held_out_fold=EXAMPLE_FOLD,
             n_tiles=int(len(f)), class_counts={ANN_NAMES[c]: int((f.label == c).sum()) for c in ANN_NAMES},
             auroc=float(roc_auc_score(t, f.p_tumour)),
             sensitivity=float((call & t).sum() / t.sum()), specificity=float((~call & ~t).sum() / (~t).sum()),
             n_training_tiles=int(len(train)), n_training_patients=int(train.patient_id.nunique()))
    m["context"] = example_context()
    return f, m


def example_context() -> dict:
    """Where the example slide sits among all held-out slides of the five UNI2 100 %-budget fold heads."""
    rows = []
    for k in range(5):
        d = HEAD_DIR.parent / f"fold{k}"
        r = pd.read_csv(d / "held_out_rows.csv.gz", dtype={"slide_id": str})
        pr = np.load(d / "probabilities.npy")
        yh = np.load(d / "predictions.npy")
        assert (r.position.to_numpy() == np.arange(len(r))).all() and pr.shape[0] == len(r)
        for sid, g in r.groupby("slide_id"):
            t = g.tissue_label.to_numpy() == TUMOUR_CLASS
            c = yh[g.position.to_numpy()] == TUMOUR_CLASS
            rows.append(dict(slide_id=sid, fold=k, n=len(g), n_tumour=int(t.sum()), n_non=int((~t).sum()),
                             auroc=float(roc_auc_score(t, pr[g.position.to_numpy(), TUMOUR_CLASS - 1])) if t.any() and (~t).any() else np.nan,
                             sensitivity=float((c & t).sum() / t.sum()) if t.any() else np.nan,
                             specificity=float((~c & ~t).sum() / (~t).sum()) if (~t).any() else np.nan))
    d = pd.DataFrame(rows)
    both = d.dropna(subset=["auroc"])
    ex = d[d.slide_id == EXAMPLE_SLIDE].iloc[0]
    return dict(n_held_out_slides=int(len(d)), n_slides_both_classes=int(len(both)),
                auroc_median=float(both.auroc.median()), auroc_q1=float(both.auroc.quantile(0.25)),
                auroc_q3=float(both.auroc.quantile(0.75)),
                specificity_median=float(d.specificity.median()), sensitivity_median=float(d.sensitivity.median()),
                example_auroc_rank=int((both.auroc > ex.auroc).sum() + 1),
                example_specificity_rank=int((d.specificity.dropna() > ex.specificity).sum() + 1),
                note="per-slide, held-out fold head of each slide's patient; ranks are 1 = best")


def load_classifier() -> pd.DataFrame:
    tm = pd.read_csv(TILE_METRICS)
    tm = tm[tm.budget_percent == 100]
    ins = pd.read_csv(IN_SAMPLE)
    ins = ins[(ins.budget_percent == 100) & (ins.scope == "all_pool_tiles")].set_index("encoder")
    ref = pd.read_csv(PANEL_DATA["A"]).set_index("encoder")
    rows = []
    for enc, _ in ENCODERS:
        pooled = tm[(tm.encoder == enc) & (tm.scope == "pooled_cross_fitted")]
        folds = tm[(tm.encoder == enc) & (tm.scope.str.match(r"^fold[0-4]$"))]
        assert len(pooled) == 1 and len(folds) == 5, enc
        p = pooled.iloc[0]
        full = json.loads(ins.loc[enc, "full_in_sample"])
        r = dict(encoder=enc, n_tiles=int(p.n),
                 sens=p.sensitivity, sens_fold_min=folds.sensitivity.min(), sens_fold_max=folds.sensitivity.max(),
                 spec=p.specificity, spec_fold_min=folds.specificity.min(), spec_fold_max=folds.specificity.max(),
                 sens_in_sample=full["sensitivity"], spec_in_sample=full["specificity"])
        x = ref.loc[enc]
        for a, b in [("sens", "sens_pooled_cross_fitted"), ("sens_fold_min", "sens_fold_min"), ("sens_fold_max", "sens_fold_max"),
                     ("spec", "spec_pooled_cross_fitted"), ("spec_fold_min", "spec_fold_min"), ("spec_fold_max", "spec_fold_max"),
                     ("sens_in_sample", "full_head_on_pool_tiles_sens"), ("spec_in_sample", "full_head_on_pool_tiles_spec")]:
            assert abs(r[a] - x[b]) < TOL, (enc, a, r[a], x[b])
        assert r["n_tiles"] == int(x.n_tiles)
        rows.append(r)
    a = pd.DataFrame(rows)
    assert a.n_tiles.tolist() == [544182, 544182, 544182, 167489]
    return a


def load_contrasts() -> pd.DataFrame:
    c = pd.read_csv(CONTRASTS)
    c = c[(c.family == "a_filtered_minus_all_tissue") & (c.level == "macro") & (c.k == 5)
          & (c.population == "both_returned") & (c.endpoint.isin(["hit", "precision"]))]
    assert len(c) == 56, len(c)  # 14 settings x 2 budgets x 2 endpoints
    assert (c.point_status == "estimable").all() and (c.ci_status == "estimable").all()
    w = pd.read_csv(WORDING)
    w = w[w.estimand == "conditional_macro_both_returned"].set_index(["encoder", "method", "arm", "budget"])
    rows = []
    for budget in (100, 5):
        ref = pd.read_csv(PANEL_DATA["B" if budget == 100 else "C"]).set_index(["encoder", "method", "arm"])
        for enc, meth, arm in SETTINGS:
            sub = c[(c.encoder == enc) & (c.method == meth) & (c.arm == arm) & (c.budget == budget)]
            h = sub[sub.endpoint == "hit"].iloc[0]
            p = sub[sub.endpoint == "precision"].iloc[0]
            r = dict(budget=budget, encoder=enc, method=meth, arm=arm, k=5,
                     hit_point=h.point, hit_ci_low=h.ci_low, hit_ci_high=h.ci_high, hit_n_queries=int(h.n_queries),
                     precision_point=p.point, precision_ci_low=p.ci_low, precision_ci_high=p.ci_high,
                     precision_n_queries=int(p.n_queries), n_patients=int(h.n_patients))
            wc = w.loc[(enc, meth, arm, budget)]
            x = ref.loc[(enc, meth, arm)]
            for col in ["hit_point", "hit_ci_low", "hit_ci_high", "precision_point", "precision_ci_low", "precision_ci_high"]:
                assert abs(r[col] - wc[col]) < TOL, ("wording_classes", enc, meth, arm, budget, col)
                assert abs(r[col] - x[col]) < TOL, ("panel_data", enc, meth, arm, budget, col)
            r["wording_class"] = wc.wording_class
            rows.append(r)
    d = pd.DataFrame(rows)
    assert len(d) == 28
    assert set(d.hit_n_queries) == {2603, 2604} and set(d.precision_n_queries) == {2603, 2604}
    assert (d.n_patients == 1800).all()
    return d


# ----------------------------------------------------------------------------- panels
def panel_example(axes, f: pd.DataFrame, m: dict) -> dict:
    """Three aligned views of the example slide: H&E, pathologist annotation, held-out P(tumor)."""
    import openslide

    sl = openslide.OpenSlide(str(WSI))
    ds = float(sl.level_downsamples[WSI_LEVEL])
    W, H = sl.level_dimensions[WSI_LEVEL]
    he = np.asarray(sl.read_region((0, 0), WSI_LEVEL, (W, H)).convert("RGB"))
    xs, ys, tl = f.x.to_numpy() / ds, f.y.to_numpy() / ds, TILE_PX / ds
    x0, x1 = max(0, int(xs.min() - 3 * tl)), min(W, int(xs.max() + 4 * tl))
    y0, y1 = max(0, int(ys.min() - 3 * tl)), min(H, int(ys.max() + 4 * tl))
    crop = he[y0:y1, x0:x1]
    ann = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    heat = np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32)
    call = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    step = int(round(tl))
    for x, y, lab, p, pr in zip(xs, ys, f.label, f.p_tumour, f.pred):
        a, b = int(round(x)) - x0, int(round(y)) - y0
        ann[b:b + step, a:a + step] = (*ANN_RGB[int(lab)], 180)
        heat[b:b + step, a:a + step] = p
        call[b:b + step, a:a + step] = int(pr == TUMOUR_CLASS)
    cmap = matplotlib.colormaps["coolwarm"]
    heat_rgba = cmap(np.nan_to_num(heat, nan=0.0))
    heat_rgba[..., 3] = np.where(np.isnan(heat), 0.0, 0.9)

    for ax in axes:
        ax.imshow(crop, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[1].imshow(ann, interpolation="none")
    axes[2].imshow(heat_rgba, interpolation="none")
    # revision (2026-11-10): no black contour around predicted-tumor tiles in panel A.
    axes[0].set_title(f"H&E, {EXAMPLE_SJID} ({EXAMPLE_GROUP_NAME})", loc="left", fontsize=7)
    axes[1].set_title("Pathologist annotation", loc="left", fontsize=7)
    axes[2].set_title("Predicted tumor probability", loc="left", fontsize=7)
    # 5 mm scale bar from level-0 mpp x actual downsample
    mpp = float(sl.properties["openslide.mpp-x"]) * ds
    bar = 5000.0 / mpp
    hh = crop.shape[0]
    axes[0].plot([12, 12 + bar], [hh - 30] * 2, color="black", linewidth=2, solid_capstyle="butt")
    axes[0].text(12, hh - 50, "5 mm", fontsize=6)
    present = [k for k in (1, 4, 3, 2, 5) if m["class_counts"][ANN_NAMES[k]] > 0]
    axes[1].legend(handles=[Patch(color=np.array(ANN_RGB[k]) / 255, label=ANN_NAMES[k]) for k in present]
                   + [Patch(facecolor="white", edgecolor="#808080", label="Not annotated")],
                   loc="upper right", fontsize=6, frameon=True, framealpha=0.85, edgecolor="none",
                   handlelength=1.0, borderaxespad=0.2, borderpad=0.3)
    sm = cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=cmap); sm.set_array([])
    axes[2].add_patch(plt.Rectangle((0.52, 0.86), 0.47, 0.14, transform=axes[2].transAxes, facecolor="white",
                                    alpha=0.85, edgecolor="none", zorder=4))
    cax = axes[2].inset_axes([0.56, 0.905, 0.38, 0.02])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cb.ax.set_title("Tumor probability", fontsize=6, pad=2, loc="left")
    return dict(wsi=str(WSI), level=WSI_LEVEL, actual_downsample=ds, level_dims=[W, H],
                crop_level_px=[x0, y0, x1, y1], mpp_x=sl.properties.get("openslide.mpp-x"),
                vendor=sl.properties.get("openslide.vendor"))


def panel_classifier(ax, a: pd.DataFrame, show_in_sample: bool = False):
    x = np.arange(len(ENCODERS))
    wide, narrow = 0.30, 0.10
    for j, (metric, color, label) in enumerate([("sens", SENS_COLOR, "Sensitivity"), ("spec", SPEC_COLOR, "Specificity")]):
        off = (-1) ** (j + 1) * 0.18
        vals = a[metric].values
        lo = vals - a[f"{metric}_fold_min"].values
        hi = a[f"{metric}_fold_max"].values - vals
        ax.bar(x + off, vals, wide, color=color, label=label, zorder=2)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.7, capsize=2, zorder=3)
        if show_in_sample:
            ax.bar(x + off + wide / 2 + narrow / 2 + 0.01, a[f"{metric}_in_sample"].values, narrow, color=GREY, zorder=2,
                   label="Training tiles" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in ENCODERS], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Proportion of annotated tiles")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc="lower left", frameon=False, handlelength=1.2, fontsize=6.5)


def panel_contrasts(axes, d: pd.DataFrame, budget: int, ylim, title: str = "",
                    settings: list = SETTINGS, metrics: tuple = ("hit", "precision")):
    """`ylim` is one (lo, hi) shared by both sub-panels, or a dict metric -> (lo, hi) when scales differ."""
    sub = d[d.budget == budget].set_index(["encoder", "method", "arm"])
    pos = []
    xi = 0.0
    prev_enc = None
    for enc, meth, arm in settings:
        if prev_enc is not None and enc != prev_enc:
            xi += 1.0  # gap between encoder blocks
        pos.append(xi)
        xi += 1.0
        prev_enc = enc
    pos = np.array(pos)
    block_centers, block_labels = [], []
    for enc, lab in ENC_LABEL_BC.items():
        idx = [i for i, s in enumerate(settings) if s[0] == enc]
        block_centers.append(pos[idx].mean())
        block_labels.append(lab)
    for ax, ep in zip(axes, metrics):
        sub_title = METRIC_LABEL[ep].replace("\n", " ")
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
        for p, (enc, meth, arm) in zip(pos, settings):
            r = sub.loc[(enc, meth, arm)]
            pt, lo, hi = r[f"{ep}_point"], r[f"{ep}_ci_low"], r[f"{ep}_ci_high"]
            color = METHOD_COLOR[meth]
            filled = arm == "both_filtered"
            if np.isnan(lo) or np.isnan(hi):  # interval not estimable (RER only): point without whisker
                ax.plot(p, pt, "o", color=color, mfc=color if filled else "white", mec=color, ms=4, zorder=3)
            else:
                ax.errorbar(p, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=color, mfc=color if filled else "white",
                            mec=color, ms=4, capsize=2, elinewidth=0.9, zorder=3)
        ax.set_xlim(pos[0] - 0.8, pos[-1] + 0.8)
        ax.set_ylim(*(ylim[ep] if isinstance(ylim, dict) else ylim))
        ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
        ax.set_xticks(block_centers)
        ax.set_xticklabels(block_labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0)
        for i in range(len(pos) - 1):
            if settings[i][0] != settings[i + 1][0]:
                ax.axvline((pos[i] + pos[i + 1]) / 2, color="#d0d0d0", linewidth=0.5, zorder=0)
        ax.text(0.02, 0.97, sub_title, transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
    axes[0].set_ylabel("Change with tumor-only filtering")
    lim0 = ylim[metrics[0]] if isinstance(ylim, dict) else ylim
    for ax, ep in list(zip(axes, metrics))[1:]:
        if (ylim[ep] if isinstance(ylim, dict) else ylim) == lim0:
            ax.set_yticklabels([])
        else:
            ax.yaxis.tick_right()
    if title:
        axes[0].set_title(title, loc="left", x=0, y=1.04)


# ----------------------------------------------------------------------------- main
def shared_ylim(d: pd.DataFrame) -> tuple[float, float]:
    """One y-scale for every filtering panel (main B and the 5 % supplement), rounded outward to 0.01."""
    lo = min(d.hit_ci_low.min(), d.precision_ci_low.min())
    hi = max(d.hit_ci_high.max(), d.precision_ci_high.max())
    return (np.floor(lo * 100) / 100, np.ceil(hi * 100) / 100)


def load_group_contrasts() -> pd.DataFrame:
    """Per-diagnostic-group version of load_contrasts() at the 100 % budget (14 groups x 14 settings x 2 endpoints).

    Drawn directly from contrasts.csv (level == group); the only prior table at group level (S-R2.1-D) covers one
    headline cell with F1/RER, so its per-group support is the available cross-check (n_queries must match)."""
    c = pd.read_csv(CONTRASTS)
    c = c[(c.family == "a_filtered_minus_all_tissue") & (c.level == "group") & (c.k == 5) & (c.budget == 100)
          & (c.population == "both_returned") & (c.endpoint.isin(["hit", "precision"]))]
    assert len(c) == 392 and c.group.nunique() == 14, len(c)
    assert (c.point_status == "estimable").all() and (c.ci_status == "estimable").all()
    sup = pd.read_csv(GROUP_SUPPORT).set_index("class")["support"]
    rows = []
    for enc, meth, arm in SETTINGS:
        for grp, sub in c[(c.encoder == enc) & (c.method == meth) & (c.arm == arm)].groupby("group"):
            h = sub[sub.endpoint == "hit"].iloc[0]
            pr = sub[sub.endpoint == "precision"].iloc[0]
            assert int(h.n_queries) in (int(sup[grp]), int(sup[grp]) - 1), (grp, h.n_queries, sup[grp])
            rows.append(dict(budget=100, encoder=enc, method=meth, arm=arm, k=5, group=grp,
                             hit_point=h.point, hit_ci_low=h.ci_low, hit_ci_high=h.ci_high, hit_n_queries=int(h.n_queries),
                             precision_point=pr.point, precision_ci_low=pr.ci_low, precision_ci_high=pr.ci_high,
                             precision_n_queries=int(pr.n_queries), n_patients=int(h.n_patients)))
    g = pd.DataFrame(rows)
    assert len(g) == 196
    return g


def add_rer_rebased(d: pd.DataFrame, g: pd.DataFrame | None, base: str = "precision"):
    """Attach RER rebased on precision@5 (or hit@5) from derive_fig3_rer_precision.py (comparison variant,
    author request 2026-10-21 ≈16:30; derived endpoint, not pre-registered)."""
    r = pd.read_csv(RER_REBASED)
    r = r[(r.base_metric == base) & (r.budget == 100)]

    def attach(frame, cls_col):
        rows = []
        for _, x in frame.iterrows():
            sub = r[(r.encoder == x.encoder) & (r.method == x.method) & (r.arm == x.arm)
                    & (r["class"] == (x[cls_col] if cls_col else "MACRO"))]
            assert len(sub) == 1 and abs(sub.iloc[0].delta - x[f"{base}_point"]) < TOL
            row = sub.iloc[0]
            rows.append(dict(rer_point=row.rer, rer_ci_low=row.rer_ci_low, rer_ci_high=row.rer_ci_high,
                             rer_ci_status=row.rer_ci_status, rer_n_undefined_reps=int(row.n_undefined_reps),
                             rer_base_metric=base))
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    d2 = attach(d[d.budget == 100], None)
    g2 = attach(g, "group") if g is not None else None
    return d2, g2


def add_rer(d: pd.DataFrame, g: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Attach the relative error reduction in top-5 F1 (the submitted Fig. 3B metric and the TF-2 pre-named headline
    cell) from rer_contrasts.csv, handling == both_returned (the population of B/C).

    RER = (F1_filtered - F1_all) / (1 - F1_all), strict plurality vote, ties = miss. Cells whose bootstrap interval is
    not estimable (> 1 % undefined replicates, F1_all ~ 1) keep their point and carry NaN interval bounds."""
    r = pd.read_csv(RER_CONTRASTS)
    r = r[(r.handling == "both_returned") & (r.rule == "plurality")]
    head = pd.read_csv(RER_CONTRASTS)
    head = head[(head.cell_role == "headline_prenamed_cell") & (head.handling == "miss") & (head["class"] == "MACRO")].iloc[0]
    assert abs(head.rer - 0.097225) < 1e-5 and head.encoder == "uni2" and head.arm == "both_filtered"
    cols = ["rer", "rer_ci_low", "rer_ci_high", "f1_all", "f1_filt", "delta_f1", "delta_ci_low", "delta_ci_high"]
    rn = {"rer": "rer_point"}

    def take(sub):
        assert len(sub) == 1, len(sub)
        row = sub.iloc[0]
        # definition check (MS L229): RER = (F1_filt - F1_all) / (1 - F1_all)
        assert abs(row.rer - (row.f1_filt - row.f1_all) / (1 - row.f1_all)) < 1e-9, row.contrast_id
        out = {rn.get(c, c): row[c] for c in cols}
        out["rer_ci_status"] = row.rer_ci_status
        out["rer_n_undefined_reps"] = int(row.n_undefined_reps)
        out["rer_n_queries"] = int(row.support)
        return out

    def attach(frame, cls_col):
        rows = []
        for _, x in frame.iterrows():
            sub = r[(r.encoder == x.encoder) & (r.method == x.method) & (r.arm == x.arm) & (r.budget == x.budget)
                    & (r["class"] == (x[cls_col] if cls_col else "MACRO"))]
            rows.append(take(sub))
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    d2 = attach(d, None)
    assert abs(d2[(d2.encoder == "uni2") & (d2.method == "ot_cosine") & (d2.arm == "both_filtered")
                  & (d2.budget == 100)].rer_point.iloc[0] - head.rer) < 1e-9  # UNI2 both_returned == miss (nothing unreturned)
    g2 = attach(g, "group") if g is not None else None
    if g2 is not None:
        assert (g2.rer_n_queries == g2.hit_n_queries).all()
    return d2, g2


def panel_groups(axes, g: pd.DataFrame, d100: pd.DataFrame, cax, settings: list = MAIN_SETTINGS,
                 metrics: tuple = ("hit", "precision")):
    """Heat maps of the filtering effect by diagnostic group (rows) and setting (columns), one per metric.

    Top row = all groups (macro, identical to panel B); dots mark cells whose 95 % interval excludes zero; a grey
    cross marks cells whose interval is not estimable (RER variant only). `cax` is one axis (shared colour scale) or a
    list of one axis per metric (own scale per metric; RER uses RER_VMAX)."""
    order = (g.groupby("group").hit_n_queries.max().sort_values(ascending=False).index.tolist())
    rows = ["All groups"] + order
    vmax = {ep: (RER_VMAX if ep == "rer" else GROUP_VMAX) for ep in metrics}
    norms = {ep: colors.TwoSlopeNorm(vmin=-vmax[ep], vcenter=0.0, vmax=vmax[ep]) for ep in metrics}
    cmap = plt.get_cmap("RdBu_r")
    ncol = len(settings)
    blocks = [(enc, lab, [j for j, s_ in enumerate(settings) if s_[0] == enc]) for enc, lab in ENC_BLOCKS]
    any_na = False
    for ax, ep in zip(axes, metrics):
        title = METRIC_LABEL[ep].replace("\n", " ")
        M = np.full((len(rows), ncol), np.nan)
        S = np.zeros_like(M, dtype=bool)
        NA = np.zeros_like(M, dtype=bool)
        for j, (enc, meth, arm) in enumerate(settings):
            mr = d100[(d100.encoder == enc) & (d100.method == meth) & (d100.arm == arm)].iloc[0]
            M[0, j] = mr[f"{ep}_point"]
            S[0, j] = mr[f"{ep}_ci_low"] > 0 or mr[f"{ep}_ci_high"] < 0
            for i, grp in enumerate(order, start=1):
                r = g[(g.encoder == enc) & (g.method == meth) & (g.arm == arm) & (g.group == grp)].iloc[0]
                M[i, j] = r[f"{ep}_point"]
                S[i, j] = r[f"{ep}_ci_low"] > 0 or r[f"{ep}_ci_high"] < 0
                NA[i, j] = np.isnan(r[f"{ep}_ci_low"]) or np.isnan(r[f"{ep}_ci_high"])
        ax.imshow(M, cmap=cmap, norm=norms[ep], aspect="auto", interpolation="nearest")
        ii, jj = np.where(S)
        ax.scatter(jj, ii, s=5, c="black", linewidths=0, zorder=3)
        ii, jj = np.where(NA)
        if len(ii):
            any_na = True
            ax.scatter(jj, ii, s=12, c="#777777", marker="x", linewidths=0.8, zorder=3)
        for _, _, idx in blocks[:-1]:
            ax.axvline(idx[-1] + 0.5, color="white", lw=1.5)
        ax.axhline(0.5, color="white", lw=2.5)
        ax.set_xticks([])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows if ax is axes[0] else [])
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", fontsize=7.5, pad=16)
        # encoder names above each block; method/arm marker under each column (same code as the shared key)
        for _, lab, idx in blocks:
            ax.text(np.mean(idx), -0.75, lab, ha="center", va="bottom", fontsize=6.5 if len(metrics) == 2 else 5.4)
        for j, (enc, meth, arm) in enumerate(settings):
            filled = arm == "both_filtered"
            ax.plot(j, len(rows) - 0.35, marker="o", ms=4, mec=METHOD_COLOR[meth], mew=1.0,
                    mfc=METHOD_COLOR[meth] if filled else "white", clip_on=False, ls="none")
        ax.set_ylim(len(rows) - 0.5 + 0.6, -0.5)
        ax.set_xlim(-0.5, ncol - 0.5)
    # one colour bar per distinct scale: hit@5/precision@5 share GROUP_VMAX, RER has RER_VMAX
    scales = []
    for ep in metrics:
        if not any(vmax[ep] == vmax[e] for e, _ in scales):
            scales.append((ep, [ep]))
        else:
            next(lst for e, lst in scales if vmax[e] == vmax[ep]).append(ep)
    caxes = cax if isinstance(cax, (list, tuple)) else [cax]
    assert len(caxes) == len(scales), (len(caxes), len(scales))
    for cx, (ep, eps) in zip(caxes, scales):
        cb = plt.colorbar(cm.ScalarMappable(norm=norms[ep], cmap=cmap), cax=cx, orientation="vertical")
        lab = ("Change with tumor-only filtering" if len(scales) == 1
               else "Change in " + " / ".join(METRIC_LABEL[e].split("\n")[0].lower() for e in eps) if len(eps) > 1
               else METRIC_LABEL[ep].replace("\n", " "))
        cb.set_label(lab, fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        v = vmax[ep]
        cb.set_ticks([-v, -v / 2, 0, v / 2, v])
    note = "\u2022 significant (95% CI excludes 0)" + ("   \u00d7 interval not estimable" if any_na else "")
    caxes[-1].text(1.0, -0.06, note, transform=caxes[-1].transAxes, ha="right", va="top", fontsize=6)


def filtering_key(fig, x: float, y: float, arms: bool = False):
    """Method by colour; with `arms`, filtering arm by fill (supplement only)."""
    handles = [plt.Line2D([], [], marker="o", color=METHOD_COLOR[k], linestyle="none", ms=4) for k in METHOD_LABEL]
    labels = list(METHOD_LABEL.values())
    if arms:
        handles += [plt.Line2D([], [], marker="o", color="black", mfc="white", linestyle="none", ms=4),
                    plt.Line2D([], [], marker="o", color="black", linestyle="none", ms=4)]
        labels += ["Tumor-only query", "Tumor-only query and library"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(x, y), ncol=len(labels), frameon=False,
               columnspacing=1.4, handletextpad=0.4)


# ---- Panel C forest (ported verbatim from fig3_panelC_candidates.py, candidate forest_rer; 2026-11-04) ----
BAND = "#f2f2f2"
C_LABEL = {"precision": "Change in precision@5 with tumor-only filtering",
           "rer": "Relative error reduction in top-5 plurality-vote F1 with tumor-only filtering"}
C_STEP = {"precision": (0.05, 0.1), "rer": (0.25, 0.5)}  # (limit rounding, tick step)
C_METHODS = ["ot_cosine", "yottixel", "titan"]


def _lims(vals_lo, vals_hi, step):
    lo = np.floor(np.nanmin(vals_lo) / step) * step
    hi = np.ceil(np.nanmax(vals_hi) / step) * step
    return float(lo), float(hi)


def forest(axes, columns, g: pd.DataFrame, d100: pd.DataFrame, xlims: dict, xlabels: dict):
    """`columns` = list of (title, [settings], metric). Rows: 'All groups' (macro from d100) + 14 groups by size."""
    order = g.groupby("group").hit_n_queries.max().sort_values(ascending=False).index.tolist()
    rows = ["All groups"] + order
    n = len(rows)
    for ci, (ax, (title, settings, ep)) in enumerate(zip(axes, columns)):
        k = len(settings)
        offs = np.linspace(-0.22, 0.22, k) if k > 1 else np.array([0.0])
        for i in range(n):
            if i % 2 == 1:
                ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        ax.axhline(0.5, color="#9a9a9a", lw=0.6, zorder=1)
        for off, (enc, meth, arm) in zip(offs, settings):
            col = METHOD_COLOR[meth]
            for i, row in enumerate(rows):
                if i == 0:
                    r = d100[(d100.encoder == enc) & (d100.method == meth) & (d100.arm == arm)].iloc[0]
                else:
                    r = g[(g.encoder == enc) & (g.method == meth) & (g.arm == arm) & (g.group == row)].iloc[0]
                pt, lo, hi = r[f"{ep}_point"], r[f"{ep}_ci_low"], r[f"{ep}_ci_high"]
                y = i + off
                if np.isnan(lo) or np.isnan(hi):  # interval not estimable (RER only)
                    ax.plot(pt, y, marker="x", color="#777777", ms=4, mew=0.9, ls="none", zorder=3)
                    continue
                sig = lo > 0 or hi < 0
                ax.plot([lo, hi], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
                ax.plot(pt, y, marker="o", ms=3.6, mec=col, mfc=col if sig else "white", mew=0.9, ls="none", zorder=3)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(rows if ci == 0 else [])
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_xlim(*xlims[ep])
        ax.grid(axis="x", lw=0.3, alpha=0.5, zorder=0)
        ax.set_title(title, fontsize=7, pad=4)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".") if v != 0 else "0"))
        if len(set(c[2] for c in columns)) > 1:  # one metric per column: label each column
            ax.set_xlabel(xlabels[ep], fontsize=6.5)
    if len(set(c[2] for c in columns)) == 1:  # one metric for all columns: one shared label
        ep = columns[0][2]
        fig = axes[0].figure
        x0 = axes[0].get_position().x0; x1 = axes[-1].get_position().x1
        fig.text((x0 + x1) / 2, axes[0].get_position().y0 - 0.028, xlabels[ep], ha="center", va="top", fontsize=6.5)
        for sp in ("left",):
            ax.spines[sp].set_visible(False)


def forest_key(fig, y: float, methods: list[str]):
    """Method by colour plus the filled / open interval rule of the forest panel C."""
    handles = [plt.Line2D([], [], marker="o", color=METHOD_COLOR[m], ls="none", ms=4) for m in methods]
    labels = [METHOD_LABEL[m] for m in methods]
    handles += [plt.Line2D([], [], marker="o", color="black", ls="none", ms=4),
                plt.Line2D([], [], marker="o", color="black", mfc="white", ls="none", ms=4)]
    labels += ["95% interval excludes zero", "interval includes zero"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.53, y), ncol=len(labels), frameon=False,
               columnspacing=1.4, handletextpad=0.4)


def panel_forest_by_encoder(fig, slot, g_main: pd.DataFrame, d_main: pd.DataFrame, metric: str = "rer") -> list:
    """Panel C as adopted 2026-11-04: one forest column per ENC_BLOCKS encoder, MAIN_SETTINGS within a column."""
    third = slot.subgridspec(1, 4, wspace=0.22, hspace=0.16)
    cols = [(lab, [s for s in MAIN_SETTINGS if s[0] == enc], metric) for enc, lab in ENC_BLOCKS]
    axes = [fig.add_subplot(third[0, i]) for i in range(4)]
    step, tick = C_STEP[metric]
    xl = _lims(np.r_[g_main[f"{metric}_ci_low"], d_main[f"{metric}_ci_low"]],
               np.r_[g_main[f"{metric}_ci_high"], d_main[f"{metric}_ci_high"]], step)
    forest(axes, cols, g_main, d_main, {metric: xl}, {metric: C_LABEL[metric]})
    for ax in axes:
        ax.set_xticks(np.arange(xl[0], xl[1] + 1e-9, tick))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) > 1e-9 else "0"))
    return axes


def main(layout: str = "ABC", metrics: str = "hit_precision_rer", rer_base: str = "f1", panel_c: str = "forest"):
    """`metrics` = "hit_precision_rer" (adopted on 2026-10-21 ≈16:10: hit@5 | precision@5 | relative
    error reduction in top-5 F1, the submitted Fig. 3B metric) or "hit_precision" (the ≈15:35 two-metric design,
    written to its own directory as an alternative). `panel_c` = "forest" (adopted on 2026-11-04: RER
    forest by group, one column per encoder) or "heatmap" (the 2026-10-21 heat maps, written to its own variant
    directory)."""
    assert panel_c in ("forest", "heatmap"), panel_c
    # The forest C draws the F1-based RER; the metric / rer_base alternatives keep their heat-map C variants.
    forest_c = layout == "ABC" and panel_c == "forest" and metrics == "hit_precision_rer" and rer_base == "f1"
    style()
    f, m = load_example()
    d = load_contrasts()
    ylim = shared_ylim(d)
    groups = load_group_contrasts() if layout == "ABC" else None
    mets = ("hit", "precision")
    if metrics == "hit_precision_rer":
        if rer_base == "f1":
            d, groups = add_rer(d, groups)
        else:
            d, groups = add_rer_rebased(d, groups, rer_base)
            METRIC_LABEL["rer"] = f"Relative error reduction\n({METRIC_LABEL[rer_base].lower()})"
        mets = ("hit", "precision", "rer")
        d100 = d[d.budget == 100]
        lo, hi = d100.rer_ci_low.min(), d100.rer_ci_high.max()
        ylim = {"hit": ylim, "precision": ylim, "rer": (np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20)}

    if layout == "AB":
        fig = plt.figure(figsize=(7.2, 6.0))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.7, 1.0], hspace=0.12, left=0.075, right=0.99, top=0.96, bottom=0.115)
    elif forest_c:  # same page and gridspec as the candidate forest_rer render (2026-11-03)
        fig = plt.figure(figsize=(7.2, 9.0))
        outer = fig.add_gridspec(4, 1, height_ratios=[1.7, 1.0, 0.22, 1.55], hspace=0.12, left=0.075, right=0.99, top=0.97, bottom=0.085)
    else:
        fig = plt.figure(figsize=(7.2, 8.8))
        outer = fig.add_gridspec(4, 1, height_ratios=[1.7, 1.0, 0.28, 1.45], hspace=0.12, left=0.075, right=0.99, top=0.97, bottom=0.07)
    top = outer[0].subgridspec(1, 3, wspace=0.06)
    axes_img = [fig.add_subplot(top[0, i]) for i in range(3)]
    bottom = outer[1].subgridspec(1, len(mets), wspace=0.12 if len(mets) == 2 else 0.16)
    axes_b = [fig.add_subplot(bottom[0, i]) for i in range(len(mets))]

    render = panel_example(axes_img, f, m)
    d_main = d[(d.budget == 100) & (d.arm == MAIN_ARM)]
    g_main = groups[groups.arm == MAIN_ARM] if groups is not None else None
    panel_contrasts(axes_b, d_main, 100, ylim, settings=MAIN_SETTINGS, metrics=mets)
    axes_img[0].text(-0.13, 1.06, "A", transform=axes_img[0].transAxes, fontsize=11, fontweight="bold", va="top")
    axes_b[0].text(-0.17, 1.08, "B", transform=axes_b[0].transAxes, fontsize=11, fontweight="bold", va="top")
    if layout == "AB":
        filtering_key(fig, 0.53, 0.005)
    elif forest_c:
        axes_c = panel_forest_by_encoder(fig, outer[3], g_main, d_main, "rer")
        axes_c[0].text(-0.17, 1.06, "C", transform=axes_c[0].transAxes, fontsize=11, fontweight="bold", va="top")
        forest_key(fig, -0.002, C_METHODS)
    else:
        if len(mets) == 3:  # hit@5 and precision@5 share one colour scale; RER has its own
            third = outer[3].subgridspec(1, 6, width_ratios=[1, 1, 0.045, 0.24, 1, 0.045], wspace=0.08)
            axes_c = [fig.add_subplot(third[0, 0]), fig.add_subplot(third[0, 1]), fig.add_subplot(third[0, 4])]
            cax = [fig.add_subplot(third[0, 2]), fig.add_subplot(third[0, 5])]
        else:
            third = outer[3].subgridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.08)
            axes_c = [fig.add_subplot(third[0, 0]), fig.add_subplot(third[0, 1])]
            cax = fig.add_subplot(third[0, 2])
        panel_groups(axes_c, g_main, d_main, cax, settings=MAIN_SETTINGS, metrics=mets)
        axes_c[0].text(-0.17, 1.14, "C", transform=axes_c[0].transAxes, fontsize=11, fontweight="bold", va="top")
        filtering_key(fig, 0.53, 0.003)

    note = (f"Panel A: slide {EXAMPLE_SLIDE} = {EXAMPLE_SJID} (Comp.Bio.ID from data/ALIdf3Ali.csv, asserted; "
            f"{EXAMPLE_GROUP}, patient {EXAMPLE_PATIENT}, patient fold {EXAMPLE_FOLD}); "
            f"H&E read with OpenSlide at level {WSI_LEVEL} (downsample {render['actual_downsample']:.3f}, crop "
            f"{render['crop_level_px']} level px); tile annotation from the pathologist label CSV (every held-out tile "
            "matched a non-zero-label row at the same level-0 coordinate); tumor probability = column 0 of "
            f"`{HEAD_PROBS.relative_to(TF2)}` on the `held_out_rows.csv.gz` rows of this slide (UNI2, 100 % budget, "
            f"fold-{EXAMPLE_FOLD} head trained on {m['n_training_tiles']:,} tiles of {m['n_training_patients']} other "
            f"patients; asserted absent from training_tiles.csv.gz). Recomputed on these {m['n_tiles']:,} tiles: "
            f"AUROC {m['auroc']:.4f}, sensitivity {m['sensitivity']:.4f}, specificity {m['specificity']:.4f} "
            f"(class counts {m['class_counts']}); only annotated tiles carry a prediction. Illustrative slide chosen by the "
            f"author (Q5(a), 2026-10-09); per-slide held-out context over {m['context']['n_slides_both_classes']} annotated "
            f"slides with both classes: median AUROC {m['context']['auroc_median']:.3f}, IQR {m['context']['auroc_q1']:.3f}-"
            f"{m['context']['auroc_q3']:.3f}; this slide ranks {m['context']['example_auroc_rank']} by AUROC and "
            f"{m['context']['example_specificity_rank']} of {m['context']['n_held_out_slides']} by specificity. "
            "Panel B: TF-2 S6 primary evaluation, family a_filtered_minus_all_tissue, macro, k = 5, population "
            "both_returned, 100 % budget, arm both_filtered only (the pre-registered headline arm; the query-only arm "
            "is drawn in S-R2.1-F4 — author, 2026-10-21 ≈15:30); n_queries 2,604 (UNI2, TITAN) or 2,603 (Virchow2, H-Optimus-1; slide 14601 not "
            "returned); 1,800 patients; whole-patient bootstrap 95 % intervals. Every value cross-checked (tolerance 1e-9) "
            "against wording_classes.csv and writing_001/fig3_panel_B_data.csv. "
            f"Y-limits {(ylim['hit'] if isinstance(ylim, dict) else ylim)[0]:.2f} to "
            f"{(ylim['hit'] if isinstance(ylim, dict) else ylim)[1]:.2f} (hit@5/precision@5) are shared with Supplementary Fig. S-R2.1-F4 (5 % budget) and come "
            "from the union of all interval bounds at both budgets. All numeric annotations, n counts and method "
            "qualifiers were deliberately removed from the figure face (author, 2026-10-21 ≈14:08) and belong in the "
            "legend. Hit-yield (full-target) estimand and wording-class letters are not drawn.")
    inputs = [HELD_OUT_ROWS, HEAD_PROBS, HEAD_PREDS, HEAD_TRAIN, LABEL_CSV, WSI, METADATA, CONTRASTS, WORDING, PANEL_DATA["B"]]
    frames = {"source_data_panelA_tiles": f, "source_data_panelB": d_main}
    figure, stem = FIGURE, "Figure_3_revised"
    if forest_c:
        note += (" Panel C (author decision, Haoran 2026-11-04, pass 013, after viewing the candidates forest_rer, "
                 "forest_prec_rer and forest_rer_hopt_yot of fig3_panelC_candidates.py; replaces the 2026-10-21 heat maps, "
                 "kept behind --panel_c=heatmap and in superseded/fig3_tumor_filtering_heatmapC_20261021.py): forest plot of "
                 "the relative error reduction in top-5 plurality-vote F1 by diagnostic group from load_group_contrasts() + "
                 "add_rer() (contrasts.csv level == group; rer_contrasts.csv both_returned), one column per tile encoder "
                 "(UNI2, Virchow2, H-Optimus-1, TITAN), Optimal Transport and Yottixel side by side within a column; rows = "
                 "macro (identical to B) then the 14 groups ordered by number of queries; point + 95 % whole-patient "
                 "bootstrap interval; filled = interval excludes zero, open = includes zero (unadjusted); grey cross = RER "
                 "interval not estimable (> 1 % of bootstrap replicates had F1_all = 1: NF and RB for H-Optimus-1 OT, RB for "
                 "TITAN), point only. x-limits from the union of drawn interval bounds rounded to 0.25, ticks every 0.5. "
                 "Per-group hit@5 and precision@5 are not drawn; they remain in Supplementary Table S-R2.1-D "
                 "(source_data_panelC_groups_all_arms.csv). Group n_queries cross-checked against "
                 "S_R2_1_D_headline_cell_groups.csv support (equal or one fewer); group hit@5 / precision@5 point estimates "
                 "are recomputed by check_fig3_panelC_groups.py (panelC_group_check.json).")
        inputs.append(GROUP_SUPPORT)
        frames["source_data_panelC_groups"] = g_main
        frames["source_data_panelC_groups_all_arms"] = groups
    elif layout == "ABC":
        figure, stem = FIGURE + "_heatmapC_variant", "Figure_3_variant_heatmapC"
        note += (" VARIANT (not the adopted render since 2026-11-04): heat-map panel C."
                 " Panel C (adopted by the author 2026-10-21 ≈14:48 after the ≈14:30 question why the submitted per-group "
                 "panel had gone): per-diagnostic-group version of B from contrasts.csv level == group (14 groups x 14 "
                 "settings x 2 endpoints, all estimable); top row = macro (identical to B); rows ordered by group size; "
                 f"dots = 95 % interval excludes zero; colour clipped at ±{GROUP_VMAX}. Group n_queries cross-checked "
                 "against S_R2_1_D_headline_cell_groups.csv support (equal or one fewer). Point estimates of all 196 x 2 "
                 "cells and the 14 macro settings are recomputed from query_metrics.csv.gz by "
                 "check_fig3_panelC_groups.py (panelC_group_check.json); intervals are not recomputed.")
        inputs.append(GROUP_SUPPORT)
        frames["source_data_panelC_groups"] = g_main
        # Both arms with intervals: feeds Table S-R2.1-D block 2 ([internal record] (8′)); the figure shows MAIN_ARM only.
        frames["source_data_panelC_groups_all_arms"] = groups
    else:
        figure, stem = FIGURE + "_AB_alternative", "Figure_3_alternative_AB"
    if metrics == "hit_precision_rer":
        inputs.append(RER_CONTRASTS)
        note += (" Third metric (author decision 2026-10-21 ≈16:10, after a hit@5 | RER comparison variant ≈15:50): "
                 "relative error reduction in top-5 F1 (RER = (F1_filt - F1_all)/(1 - F1_all), strict plurality vote, "
                 "ties = miss; definition asserted on every row) from rer_contrasts.csv, handling both_returned (same "
                 "population as hit/precision); the UNI2 OT both-sided 100 % cell equals the pre-named headline "
                 "(+0.097 [+0.017, +0.175]). RER has its own y-scale in B"
                 + (" and is the only metric of the forest C (own x-scale)" if forest_c else
                    f" and its own colour scale (±{RER_VMAX}; RB / H-Optimus-1 OT -0.78 clipped) in C")
                 + ". Cells whose interval is not estimable (> 1 % "
                 "of bootstrap replicates had F1_all = 1, denominator 0: RB and NF for H-Optimus-1 OT, RB for TITAN) "
                 "show the point with a grey cross and no whisker; their delta-F1 intervals all include zero.")
    else:
        figure, stem = FIGURE + "_hit_precision_alternative", "Figure_3_alternative_hit_precision"
    if rer_base != "f1":
        figure, stem = FIGURE + f"_rer_{rer_base}_variant", f"Figure_3_variant_RER_{rer_base}"
        inputs.append(RER_REBASED)
        note += (f" VARIANT (author request 2026-10-21 ≈16:30; not the adopted render): RER rebased on {rer_base}@5 — "
                 f"RER = (metric_filt - metric_all)/(1 - metric_all) on the group mean of conditional {rer_base}@5, "
                 "population both_returned, intervals from the S6 shared patient draws (derive_fig3_rer_precision.py; "
                 "derived endpoint, not pre-registered; the F1-based RER remains the pre-named headline).")
    out = save(fig, figure, stem, inputs, note, frames)
    metrics = out / "panelA_metrics.json"
    metrics.write_text(json.dumps({**m, "render": render}, indent=1) + "\n")
    with open(out / "SHA256SUMS.txt", "a") as fh:
        fh.write(f"{sha256(metrics)}  {metrics.name}\n")


if __name__ == "__main__":
    main("AB" if "--layout=AB" in sys.argv[1:] else "ABC",
         "hit_precision" if "--metrics=hit_precision" in sys.argv[1:] else "hit_precision_rer",
         "precision" if "--rer_base=precision" in sys.argv[1:] else "hit" if "--rer_base=hit" in sys.argv[1:] else "f1",
         "heatmap" if "--panel_c=heatmap" in sys.argv[1:] else "forest")
