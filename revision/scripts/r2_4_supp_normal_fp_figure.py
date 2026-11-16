"""R2.4-T5: Supplementary figure on normal-tissue false positives of the tile classifier.

Design decision 2026-10-21 (R2.4-d = yes; hilar region stated factually, option (b)).
Figure assembly only: no new analysis. Inputs are the accepted S3 per-slide table and the two
existing S3 renders; panel C is drawn from the table.

  A  SJNORM059662_O1 (internal scan key 10113) with the tile classifier's calls (red = called tumor).
  B  The same slide (query) and its five nearest reference slides under the primary
     configuration (UNI2 + Optimal Transport, all-tissue, same-patient slides excluded).
  C  Per-slide fraction of mosaic tiles called tumor, by tumor group: the 2,689 non-MISC slides plotted,
     of the 2,825 slides with a mosaic tile count overall (title wording corrected in revision, [internal record]
     SI unit 10; no point or count changed); SJNORM059662_O1 marked.

Decision D1 (2026-11-04): the face shows de-identified slide IDs (Comp.Bio.ID, via
scripts/figures/_common.SCAN_TO_SLIDE_ID, verified against data/ALIdf3Ali.csv by
scripts/figures/scan_to_slide_id.py), not numeric scan keys. The embedded S3 neighbours image prints a
small scan-ID caption in a white band at the top-left of each cell (rows 0-22, columns 0-420 of each
900 x 832 cell); that band is blanked to white before the overlay labels are drawn.

Usage:  export PROSPER_OUT=...; python3 scripts/r2_4_supp_normal_fp_figure.py [--out <dir>]
Output: $PROSPER_OUT/r2_4_supp_normal_fp_20261021/ (or <dir>) {supp_fig_normal_fp.png,.pdf,panel_c_summary.csv,
        record.json,SHA256SUMS.txt}
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
from _common import slide_id  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("PROSPER_OUT", ROOT / "prosper_revision_output"))
S3_CSV = OUT_ROOT / "s3_tile_classifier_tumour_fraction.csv"
PANEL_A = ROOT / "docs/results/s3/sjnorm059662_tile_classifier_calls.png"
PANEL_B = ROOT / "docs/results/s3/sjnorm059662_neighbours.png"
OUT = OUT_ROOT / "r2_4_supp_normal_fp_20261021"
QUERY = "10113"
# Cell labels for panel B, left-to-right, top-to-bottom, as rendered in the S3 neighbours image
# (query, then ranks 1-5; identities and labels verified from the ranked lists on 2026-10-21).
B_SCANS = ["10113", "17284", "15182", "7899", "10108", "10111"]
B_CELLS = [  # two lines so the slide IDs fit the cell width
    f"Query: {slide_id('10113')}\nnormal kidney",
    f"Rank 1: {slide_id('17284')}\nWT",
    f"Rank 2: {slide_id('15182')}\nWT",
    f"Rank 3: {slide_id('7899')}\nnormal kidney",
    f"Rank 4: {slide_id('10108')}\nWT",
    f"Rank 5: {slide_id('10111')}\nnormal kidney",
]
QUERY_ID = slide_id(QUERY)
# White band holding the baked-in scan-ID caption of each cell of PANEL_B (measured 2026-11-04:
# pure-white rows 0-22, columns 0-420 in every one of the six 900 x 832 cells).
B_CAPTION_BAND = (23, 421)
GROUP_ORDER = ["NORMAL", "RB", "MEL", "LIVER", "OS", "WT", "ACT/ACC", "SS", "NF", "MPNST",
               "EWS", "DSRCT", "MRT", "NB", "RMS"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    global OUT
    if len(sys.argv) > 2 and sys.argv[1] == "--out":  # revision: write a new render elsewhere
        OUT = Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    s3 = pd.read_csv(S3_CSV)
    s3["sid"] = s3["Slide.Scan.File"].astype(str)
    n_with_fraction = int(s3["frac"].notna().sum())
    assert n_with_fraction == 2825, n_with_fraction
    s3 = s3[s3["frac"].notna() & (s3["Tumor.group"] != "MISC")].copy()
    assert len(s3) == 2689, len(s3)
    groups = [g for g in GROUP_ORDER if g in set(s3["Tumor.group"])]
    assert len(groups) == 15, groups
    q = s3[s3.sid == QUERY]
    assert len(q) == 1 and q["Tumor.group"].iloc[0] == "NORMAL", "10113 must be a NORMAL slide"
    q_frac = float(q["frac"].iloc[0])

    # panel C summary
    summ = (s3.groupby("Tumor.group")["frac"]
            .agg(n="size", mean="mean", median="median",
                 q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
            .reindex(groups).reset_index())
    summ.to_csv(OUT / "panel_c_summary.csv", index=False)

    img_a = Image.open(PANEL_A).convert("RGB")
    img_b = Image.open(PANEL_B).convert("RGB")
    arr_b = np.asarray(img_b).copy()
    wb, hb = img_b.size
    assert (wb, hb) == (2700, 1664), (wb, hb)
    for i in range(6):
        r, c = divmod(i, 3)
        y0, x0 = int(r * hb / 2), int(c * wb / 3)
        arr_b[y0:y0 + B_CAPTION_BAND[0], x0:x0 + B_CAPTION_BAND[1]] = 255
    img_b = Image.fromarray(arr_b)

    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0], width_ratios=[1.0, 1.25],
                          hspace=0.28, wspace=0.06)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[1, :])

    ax_a.imshow(img_a); ax_a.set_axis_off()
    ax_a.set_title(f"A  {QUERY_ID} (normal kidney): tile-classifier calls\n"
                   "red = called tumor (906 of 2,008 mosaic tiles, 45%); blue = not called",
                   loc="left", fontsize=9)

    ax_b.imshow(img_b); ax_b.set_axis_off()
    ax_b.set_title(f"B  Query {QUERY_ID} and its five nearest reference slides\n"
                   "(UNI2 + Optimal Transport, all tissue; same-patient slides excluded; WT = Wilms tumor)",
                   loc="left", fontsize=9)
    w, h = img_b.size
    cw, ch = w / 3, h / 2
    for i, label in enumerate(B_CELLS):
        r, c = divmod(i, 3)
        ax_b.text(c * cw + 10, r * ch + 12, label, fontsize=7, va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.3", alpha=0.9))

    data = [s3.loc[s3["Tumor.group"] == g, "frac"].to_numpy() for g in groups]
    bp = ax_c.boxplot(data, positions=range(len(groups)), widths=0.6, showfliers=False,
                      patch_artist=True, medianprops=dict(color="black"))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor("#d9534f" if g == "NORMAL" else "#dddddd")
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(groups, data)):
        ax_c.scatter(i + rng.uniform(-0.22, 0.22, len(vals)), vals, s=4, color="0.35", alpha=0.35,
                     linewidths=0, zorder=2)
    i_norm = groups.index("NORMAL")
    ax_c.scatter([i_norm], [q_frac], s=70, marker="D", color="#d9534f", edgecolor="black", zorder=5)
    ax_c.annotate(f"{QUERY_ID}\n(45%)", (i_norm, q_frac), xytext=(i_norm + 0.38, 0.27), fontsize=8.5,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                  arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax_c.set_xlim(-0.7, len(groups) - 0.3)
    ax_c.set_xticks(range(len(groups)))
    ax_c.set_xticklabels([f"{g}\n(n={int(summ.loc[summ['Tumor.group'] == g, 'n'].iloc[0])})" for g in groups],
                         fontsize=8.5)
    ax_c.set_ylabel("Fraction of mosaic tiles called tumor", fontsize=10)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_title(f"C  Per-slide tumor-call fraction by tumor group ({len(s3):,} non-MISC slides plotted; "
                   f"{n_with_fraction:,} slides with a mosaic tile count overall;\n"
                   "boxes = median and interquartile range, whiskers = 1.5 x IQR, points = slides)",
                   loc="left", fontsize=10)
    ax_c.grid(axis="y", lw=0.4, alpha=0.5)

    png = OUT / "supp_fig_normal_fp.png"; pdf = OUT / "supp_fig_normal_fp.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    normal = s3[s3["Tumor.group"] == "NORMAL"]["frac"]
    record = {
        "script": str(Path(__file__).relative_to(ROOT)), "script_sha256": sha256(Path(__file__)),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "matplotlib": matplotlib.__version__, "pandas": pd.__version__,
        "inputs": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha256(p)
                   for p in (S3_CSV, PANEL_A, PANEL_B)},
        "population": {"slides_with_fraction_non_MISC": int(len(s3)), "groups": groups},
        "normal": {"n": int(len(normal)), "median": round(float(normal.median()), 4),
                   "mean": round(float(normal.mean()), 4), "lt_5pct": int((normal < 0.05).sum()),
                   "gt_10pct": int((normal > 0.10).sum()), "gt_50pct": int((normal > 0.50).sum())},
        "query_10113": {"frac": round(q_frac, 4), "n_tumour": int(q["n_tumour"].iloc[0]),
                        "n_mosaic": int(q["n_mosaic"].iloc[0]),
                        "rank_among_normal": int((normal > q_frac).sum()) + 1},
        "panel_b_cells": B_CELLS,
        "panel_b_scan_keys": B_SCANS,
        "panel_b_caption_band_blanked_rows_cols": list(B_CAPTION_BAND),
        "slide_id_map": "scripts/figures/_common.SCAN_TO_SLIDE_ID (decision D1, 2026-11-04)",
        "author_decision": "2026-10-21 Haoran: R2.4-d yes; hilar region stated factually (option b)",
    }
    (OUT / "record.json").write_text(json.dumps(record, indent=2))
    with open(OUT / "SHA256SUMS.txt", "w") as fh:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.name != "SHA256SUMS.txt":  # skip superseded_*/ archive folders
                fh.write(f"{sha256(p)}  {p.name}\n")
    print(json.dumps(record["normal"]), json.dumps(record["query_10113"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
