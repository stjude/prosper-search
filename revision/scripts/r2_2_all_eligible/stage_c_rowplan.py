#!/usr/bin/env python3
"""R2.2-ALL Stage C: OT row-task plan from the C3 pair-probe calibration.

Fits solve+cost time = exp(a) * cells^p and peak RSS = k * cells + c on the accepted cap-1e8 CPU
pair probes (>= 1e6 cells), then, for each OT-row query, sorts the 2,777 other bank slides by
size, assigns each reference to the memory tier of its pair cells, and packs consecutive refs
of one tier into tasks of about --target-h predicted hours. Self/same-patient refs stay in the
lists (the row worker records them as excluded without solving).

Writes <out>/refs/<query>_<tag>.txt, <out>/tasks_<tier>.txt (one wrapper command per task) and
<out>/rowplan.json (fit, tiers, per-task predicted hours / max cells).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import stage_c_lib as L  # noqa: E402

# (max cells, RAM GB): RAM >= 1.35 x the fitted peak RSS at the tier's upper bound
TIERS = [(100e6, 8), (350e6, 24), (700e6, 48), (1.1e9, 64)]


def fit(pairs_dir):
    P = [json.loads(Path(f).read_text()) for f in glob.glob(str(Path(pairs_dir) / "pair_*_cap1e08.json"))]
    P = [p for p in P if p.get("status") == "accepted" and p["cells"] >= 1e6]
    c = np.array([p["cells"] for p in P], float)
    t = np.array([p["t_solve"] + p["t_cost"] for p in P], float)
    rss = np.array([p["vmhwm_after_solve_gb"] for p in P], float)
    p_, a_ = np.polyfit(np.log(c), np.log(t), 1)
    k, c0 = np.polyfit(c, rss, 1)
    return {"n": len(P), "exponent": float(p_), "log_intercept": float(a_),
            "resid_max": float(np.abs(np.log(t) - (a_ + p_ * np.log(c))).max()),
            "rss_gb_per_gcell": float(k * 1e9), "rss_gb_intercept": float(c0)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(L.STAGE_C / "c3/pairs"))
    ap.add_argument("--out", default=str(L.STAGE_C / "c3/rowplan"))
    ap.add_argument("--rows-out", default=str(L.STAGE_C / "c3/rows"))
    ap.add_argument("--code", default=str(L.STAGE_C / "code_v3"))
    ap.add_argument("--target-h", type=float, default=2.0)
    a = ap.parse_args(argv)
    import pandas as pd
    f = fit(a.pairs)
    for mx, gb in TIERS:
        need = f["rss_gb_per_gcell"] * mx / 1e9 + f["rss_gb_intercept"]
        if gb < 1.35 * need:
            raise ValueError(f"tier {mx:.3g} cells: {gb} GB < 1.35 x fitted {need:.1f} GB")
    sizes = pd.read_csv(L.STAGE_B / "bank_per_slide.csv", dtype={"slide": str}).set_index("slide")["n_rows"]
    slides = L.bank_slides()
    out = Path(a.out)
    (out / "refs").mkdir(parents=True, exist_ok=True)
    tasks = {gb: [] for _, gb in TIERS}
    plan = []
    for q in L.selection()["ot_rows"]:
        refs = sorted((s for s in slides if s != q), key=lambda s: (sizes[s], int(s)))
        groups = {}
        for r in refs:
            cells = float(sizes[q]) * float(sizes[r])
            gb = next(g for mx, g in TIERS if cells <= mx)
            groups.setdefault(gb, []).append((r, cells))
        for gb, items in groups.items():
            chunk, hrs = [], 0.0
            chunks = []
            for r, cells in items:
                h = np.exp(f["log_intercept"]) * cells ** f["exponent"] / 3600
                if chunk and hrs + h > a.target_h:
                    chunks.append((chunk, hrs))
                    chunk, hrs = [], 0.0
                chunk.append((r, cells))
                hrs += h
            if chunk:
                chunks.append((chunk, hrs))
            for i, (ch, hrs) in enumerate(chunks):
                tag = f"m{gb}_{i:03d}"
                rf = out / "refs" / f"{q}_{tag}.txt"
                rf.write_text("\n".join(r for r, _ in ch) + "\n")
                tasks[gb].append(f"bash {a.code}/stage_c_task.sh 1 0 stage_c_ot.py row --query {q} "
                                 f"--refs {rf} --tag {tag} --cap 1e8 --out {a.rows_out}")
                plan.append({"query": q, "tag": tag, "ram_gb": gb, "n_refs": len(ch), "pred_h": hrs,
                             "max_cells": max(c for _, c in ch)})
    for gb, cmds in tasks.items():
        if cmds:
            (out / f"tasks_m{gb}.txt").write_text("\n".join(cmds) + "\n")
    L.dump_json(out / "rowplan.json", {"fit": f, "tiers": TIERS, "target_h": a.target_h, "tasks": plan})
    by = {}
    for t in plan:
        b = by.setdefault(t["ram_gb"], [0, 0.0, 0.0])
        b[0] += 1
        b[1] += t["pred_h"]
        b[2] = max(b[2], t["pred_h"])
    print(json.dumps({"fit": f, "tiers_tasks_predh_maxtaskh": by}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
