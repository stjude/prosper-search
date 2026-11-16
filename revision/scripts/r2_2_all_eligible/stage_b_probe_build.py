#!/usr/bin/env python3
"""R2.2-ALL B2: build the pre-declared fidelity probe set (seed 20261113) from Stage A manifests.

P1 subsample: the 3 fallback slides + 3 slides per class from the other target slides; per slide
up to 24 mosaic tiles with recon coverage 1.0 and up to 24 with coverage < 1.0 (the short stratum
is filled from the other), written in grid row-major order.
P2 replay: full mosaic of 15678 and 16973 in simplified (mosaic-bank) order.
P3 throughput: all D1 gap tiles of the non-fallback gap slide whose gap count is closest to 4,000.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_b_lib as L  # noqa: E402

SEED = 20261113
LABELS = L.H / "comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv"
REPLAY = ("15678", "16973")
PER_STRATUM = 24
PER_CLASS = 3
P3_TARGET = 4000


def manifest_path(stage_a, slide):
    d = "slides_option_i" if slide in L.FALLBACK_SLIDES else "slides"
    return Path(stage_a) / d / f"{slide}.manifest.csv.gz"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sa = Path(args.stage_a)
    out = Path(args.out)
    for p in ("p1", "p2", "p3"):
        (out / p).mkdir(parents=True, exist_ok=False)

    target = [s.strip() for s in (sa / "inputs/target_slides.txt").read_text().split() if s.strip()]
    lab = pd.read_csv(LABELS, dtype=str)
    cls = dict(zip(lab["Slide.Scan.File"], lab["Tumor.group"]))
    rng = np.random.default_rng(SEED)
    others = sorted((s for s in target if s not in L.FALLBACK_SLIDES), key=int)
    by_class = {}
    for s in others:
        by_class.setdefault(cls[s], []).append(s)
    p1_slides = list(L.FALLBACK_SLIDES)
    for c in sorted(by_class):
        pool = by_class[c]
        pick = rng.choice(len(pool), size=min(PER_CLASS, len(pool)), replace=False)
        p1_slides += [pool[i] for i in sorted(pick)]

    rec = {"seed": SEED, "per_stratum": PER_STRATUM, "per_class": PER_CLASS,
           "labels": str(LABELS), "labels_sha256": L.sha256_file(LABELS), "p1": {}, "p2": {}, "p3": {}}
    n1 = 0
    for s in p1_slides:
        df = L.read_manifest(manifest_path(sa, s))
        mos = df[df["in_mosaic"] == 1]
        full = mos[mos["coverage"] == 1.0]
        part = mos[mos["coverage"] < 1.0]
        srng = np.random.default_rng([SEED, int(s)])
        kp = min(PER_STRATUM, len(part))
        kf = min(len(full), 2 * PER_STRATUM - kp)
        kp = min(len(part), 2 * PER_STRATUM - kf)
        sel = pd.concat([full.iloc[np.sort(srng.choice(len(full), kf, replace=False))] if kf else full.iloc[:0],
                         part.iloc[np.sort(srng.choice(len(part), kp, replace=False))] if kp else part.iloc[:0]])
        sel = sel.sort_values(["grid_i", "grid_j"])
        xy = sel[["wsi_loc_x", "wsi_loc_y"]].to_numpy()
        # every probe tile must exist in the mosaic bank
        _, bxy = L.load_mosaic_bank(s)
        L.match_rows(bxy, xy)
        sha = L.write_coords(out / "p1" / f"{s}.coords.csv.gz", xy, sel["coverage"].to_numpy())
        rec["p1"][s] = {"class": cls[s], "n": int(len(sel)), "n_full": int(kf), "n_partial": int(kp),
                        "coords_sha256": sha}
        n1 += len(sel)
    for s in REPLAY:
        _, bxy = L.load_mosaic_bank(s)
        df = L.read_manifest(manifest_path(sa, s))
        cov = df.set_index(L.coord_key(df[["wsi_loc_x", "wsi_loc_y"]].to_numpy()))["coverage"]
        c = cov.loc[L.coord_key(bxy)].to_numpy()
        sha = L.write_coords(out / "p2" / f"{s}.coords.csv.gz", bxy, c)
        rec["p2"][s] = {"n": int(len(bxy)), "coords_sha256": sha}
    gap = pd.read_csv(sa / "gap_manifest.csv.gz", float_precision="round_trip", dtype={"slide": str})
    gap = gap[~gap["slide"].isin(L.FALLBACK_SLIDES)]
    cnt = gap.groupby("slide").size()
    d = (cnt - P3_TARGET).abs()
    p3 = sorted(d[d == d.min()].index, key=int)[0]
    g = gap[gap["slide"] == p3]
    # cross-check against the slide manifest under D1
    df = L.read_manifest(manifest_path(sa, p3))
    m = df[L.gap_mask(df)]
    assert np.array_equal(np.sort(L.coord_key(m[["wsi_loc_x", "wsi_loc_y"]].to_numpy())),
                          np.sort(L.coord_key(g[["wsi_loc_x", "wsi_loc_y"]].to_numpy())))
    m = m.sort_values(["grid_i", "grid_j"])
    sha = L.write_coords(out / "p3" / f"{p3}.coords.csv.gz", m[["wsi_loc_x", "wsi_loc_y"]].to_numpy(),
                         m["coverage"].to_numpy())
    rec["p3"][p3] = {"n": int(len(m)), "coords_sha256": sha}
    rec["summary"] = {"p1_slides": len(p1_slides), "p1_tiles": n1,
                      "p1_full": sum(v["n_full"] for v in rec["p1"].values()),
                      "p1_partial": sum(v["n_partial"] for v in rec["p1"].values()),
                      "p1_classes": len({v["class"] for v in rec["p1"].values()}),
                      "p2_tiles": sum(v["n"] for v in rec["p2"].values()), "p3_slide": p3,
                      "p3_tiles": int(len(m))}
    for p, ss in (("p1", p1_slides), ("p2", list(REPLAY)), ("p3", [p3])):
        (out / f"{p}_slides.txt").write_text("\n".join(ss) + "\n")
    L.dump_json(out / "probe_set.json", rec)
    print(json.dumps(rec["summary"]))


if __name__ == "__main__":
    main()
