#!/usr/bin/env python3
"""R3.5-TCM: tumour-correspondence slide map ([analysis plan], pre-registered 2026-11-11).

For each reference mosaic tile, the minimum UNI2 cosine distance to the query's
classifier-predicted tumour tiles; projected to the patch grid by the mean over owning
tiles, min-max normalized and inverted, scored by variant-A AUROC (tumour vs annotated
non-tumour). Predicted tumour = TF-2 UNI2 b100 mask of the head held out for the
*reference's* patient (``fold{slide_fold(reference)}_mask.npy``), so no reference label
reaches the map; sensitivity: the query's own assigned head (``full_mask.npy``). Fewer
than 8 predicted tumour tiles -> all query tiles (counted). Compared on the same pairs
and the same whole-patient draws with the saved transport matched-cost map (P2 only),
the unconstrained nearest-tile map and the cosine ROI heat map. Read-only inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import r3_5_l1_maps as L  # noqa: E402

OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
OUTD = OUT / "r3_5_tumour_correspondence_map_20261111"
TF2 = OUT / "r2_1_filter_tf2_20261016/s3/uni2/b100"
C4_ROI8 = OUT / "c4_localization_roi8/per_pair_c4_localization_roi8.csv"
MIN_TUMOUR_TILES = 8
MAPS = ["tcm", "tcm_own", "nn1", "roi_cos", "ot"]
PRIMARY_CONTRASTS = [("tcm", "ot"), ("tcm", "roi_cos"), ("tcm", "nn1"), ("tcm_own", "tcm")]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def score(ref, tile_cost, LC):
    h = L.normalise_invert(L.project(tile_cost, ref.owners, "mean"))
    y_t = ref.patch_label == L.TUMOR
    keep = y_t | np.isin(ref.patch_label, L.NONTUMOR)
    if y_t[keep].sum() == 0 or (~y_t[keep]).sum() == 0:
        return np.nan
    return LC._auroc(h[keep], y_t[keep].astype(int))


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    LC, _ = L.import_prosper()
    s0 = L.latest_complete("s0_pairs")
    bind = Path(json.loads((s0 / "COMPLETE.json").read_text())["s0_bind"])
    meta = L.meta_cache_load(bind / "meta_tiles.pkl")
    p1 = pd.read_csv(s0 / "pairs_P1.csv", dtype=str, keep_default_na=False)
    p2 = pd.read_csv(s0 / "pairs_P2.csv", dtype=str, keep_default_na=False)
    off = pd.read_csv(TF2 / "slide_offsets.csv", dtype={"slide_id": str}).set_index("slide_id")
    masks = {f"fold{k}": np.load(TF2 / f"fold{k}_mask.npy", mmap_mode="r") for k in range(5)}
    masks["full"] = np.load(TF2 / "full_mask.npy", mmap_mode="r")
    coords = np.load(TF2 / "coordinates.npy", mmap_mode="r")
    tc = pd.read_csv(L.ROOT / "l2" / "run_003" / "per_tile_cost_D2.csv.gz", dtype={"reference": str, "query": str})
    tc = {k: g.sort_values("ref_row") for k, g in tc.groupby(["reference", "query"])}
    c4 = pd.read_csv(C4_ROI8, dtype={"reference": str})
    refs44 = set(c4[c4["auroc_A"].notna()]["reference"])
    inputs = {str(p): sha(p) for p in [s0 / "pairs_P1.csv", s0 / "pairs_P2.csv", TF2 / "slide_offsets.csv",
                                       TF2 / "full_mask.npy", TF2 / "coordinates.npy", C4_ROI8,
                                       L.ROOT / "l2" / "run_003" / "per_tile_cost_D2.csv.gz"]
              + [TF2 / f"fold{k}_mask.npy" for k in range(5)]}

    qcache = {}

    def query(q, mosaic):
        if q not in qcache:
            Fq, fc = L.load_features(q)
            a0, a1 = int(off.at[q, "tile_start"]), int(off.at[q, "tile_stop"])
            if a1 - a0 != len(fc) or not np.array_equal(np.asarray(coords[a0:a1]), np.asarray(fc, dtype=np.int64)):
                raise RuntimeError(f"{q}: TF-2 rows differ from the feature rows")
            lab = L.load_labels(q) if (L.LABELS / f"{q}.csv").exists() else {}
            qcache[q] = dict(F=unit(Fq), Fraw=Fq, a0=a0, a1=a1,
                             lab=np.array([lab.get(c, 0) for c in fc], dtype=np.int8))
        if mosaic and "X" not in qcache[q]:
            qcache[q]["X"] = LC.Slide(q, L.FEATURES, meta[q], L.LABELS).X
        return qcache[q]

    rows = []
    for design, pairs in (("P2", p2), ("P1", p1)):
        for r in L.numeric_sorted(set(pairs["reference"])):
            ref = L.MapSlide(r, meta[r], LC=LC)
            Xr = unit(ref.slide.X)
            f_ref = int(off.at[r, "slide_fold"])
            for _, row in pairs[pairs["reference"] == r].iterrows():
                Q = query(row["query"], mosaic=design == "P2")
                D = 1.0 - Xr @ Q["F"].T
                rec = dict(design=design, pair_id=row["pair_id"], reference=r, ref_patient=row["ref_patient"],
                           ref_group=row["ref_group"], ref_strict=row["ref_strict"] == "True",
                           in_manuscript_44=(design == "P2" and r in refs44), half=row["half"], arm=row["arm"],
                           query=row["query"], ref_fold=f_ref)
                for name, key in (("tcm", f"fold{f_ref}"), ("tcm_own", "full")):
                    m = np.asarray(masks[key][Q["a0"]:Q["a1"]])
                    rec[f"n_pred_tumour_{name}"] = int(m.sum())
                    rec[f"fallback_{name}"] = bool(m.sum() < MIN_TUMOUR_TILES)
                    cost = D.min(axis=1) if m.sum() < MIN_TUMOUR_TILES else D[:, m].min(axis=1)
                    rec[f"auroc_{name}"] = score(ref, cost, LC)
                    sel = m & (Q["lab"] > 0)
                    if sel.any():
                        rec[f"pred_tumour_precision_annotated_{name}"] = float((Q["lab"][sel] == L.TUMOR).mean())
                rec["auroc_nn1"] = score(ref, D.min(axis=1), LC)
                if design == "P2":  # C4 ROIs index the query's Slide.X; P1 ROIs index its feature rows
                    idx, src = L.parse_idx(row["query_tiles"]), Q.get("X")
                else:
                    kind = row["roi_primary"]
                    idx = L.parse_idx(row[f"roi_{kind}"]) if kind in ("labelled", "predicted") else None
                    src = Q["Fraw"]
                if idx is not None and len(idx):
                    qv = src[idx].mean(axis=0)
                    rec["auroc_roi_cos"] = score(ref, L.roi_distances(ref.slide.X, qv, "cosine"), LC)
                g = tc.get((r, row["query"])) if design == "P2" else None
                if g is not None and len(g) == len(ref.F):
                    rec["auroc_ot"] = score(ref, g["cost"].to_numpy()[ref.rows], LC)
                rows.append(rec)
            print(f"[tcm] {design} {r} done ({len(rows)} pairs, {time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTD / "per_pair.csv.gz", index=False)
    acc = pd.read_csv(L.ROOT / "s2" / "run_003" / "per_pair_A.csv.gz", usecols=["pair_id", "setting", "roi_primary", "auroc_A"])
    acc = acc[(acc["setting"] == "A|cosine|mean") & acc["roi_primary"]].set_index("pair_id")["auroc_A"]
    j = df.set_index("pair_id")["auroc_roi_cos"].dropna()
    j = pd.concat([j, acc.reindex(j.index)], axis=1, keys=["new", "accepted"]).dropna()
    fidelity = dict(pairs_compared=int(len(j)), pairs_with_roi_map=int(df["auroc_roi_cos"].notna().sum()),
                    max_abs_diff_vs_accepted_cosine_roi=float((j["new"] - j["accepted"]).abs().max()))
    fidelity["pass"] = fidelity["pairs_compared"] == fidelity["pairs_with_roi_map"] and \
        fidelity["max_abs_diff_vs_accepted_cosine_roi"] <= 1e-9
    print("fidelity", fidelity, flush=True)

    sets = {"P2_manuscript44": (df["design"] == "P2") & df["in_manuscript_44"],
            "P2_all46": df["design"] == "P2", "P1_all": df["design"] == "P1"}
    summ, paired = [], []
    for sname, smask in sets.items():
        for arm in ("same_group", "other_group"):
            for half in ("development", "report", "pooled"):
                x = df[smask & (df["arm"] == arm) & ((df["half"] == half) if half != "pooled" else True)]
                if x.empty:
                    continue
                for m in MAPS:
                    c = f"auroc_{m}"
                    if c not in x or x[c].notna().sum() == 0:
                        continue
                    pp = x.dropna(subset=[c]).groupby(["ref_patient", "reference"])[c].mean().groupby("ref_patient").mean()
                    ci = L.whole_patient_ci(pp.to_numpy(), list(pp.index))
                    summ.append(dict(set=sname, arm=arm, half=half, map=m, n_pairs=int(x[c].notna().sum()), **ci))
                for a_, b_ in PRIMARY_CONTRASTS:
                    ca, cb = f"auroc_{a_}", f"auroc_{b_}"
                    if ca not in x or cb not in x:
                        continue
                    y = x.dropna(subset=[ca, cb])
                    if y.empty:
                        continue
                    d = (y[ca] - y[cb]).groupby([y["ref_patient"], y["reference"]]).mean().groupby(level=0).mean()
                    ci = L.whole_patient_ci(d.to_numpy(), list(d.index))
                    paired.append(dict(set=sname, arm=arm, half=half, minuend=a_, subtrahend=b_, n_pairs=len(y), **ci))
    S, P = pd.DataFrame(summ), pd.DataFrame(paired)
    S.to_csv(OUTD / "summary.csv", index=False)
    P.to_csv(OUTD / "paired_differences.csv", index=False)
    fb = df.groupby("design")[["fallback_tcm", "fallback_tcm_own"]].sum().to_dict()
    rec = dict(fidelity=fidelity, inputs=inputs, script=str(Path(__file__).resolve()),
               script_sha256=sha(Path(__file__).resolve()),
               n_pairs=df.groupby("design").size().to_dict(), fallbacks=fb, min_tumour_tiles=MIN_TUMOUR_TILES,
               manuscript_44_refs=len(refs44), seconds=round(time.time() - t0, 1),
               created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (OUTD / "run.json").write_text(json.dumps(rec, indent=1, default=str))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        s = S[S["half"] == "pooled"].pivot_table(index=["set", "arm"], columns="map", values="point")
        print(s.round(3))
        print(P[(P["half"] == "pooled")][["set", "arm", "minuend", "subtrahend", "point", "ci95_lo", "ci95_hi"]].round(3))
        print("fallbacks:", fb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
