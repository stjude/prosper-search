#!/usr/bin/env python3
"""R3.5-ROI6 production task list: per configuration, query slides in numeric order grouped
by cumulative tile count; every task first reruns the sentinel slide against its A100
canary reference (gate G3). Cosine configurations are listed first."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
ROOT = OUT / "roi_search_6cfg_20261111"
PY = "/HOME/miniconda3/bin/python3"
HARNESS = REPO / "scripts/roi_search_6cfg/roi_search_gpu.py"
CONFIGS = [("uni2_cos", "uni2", "cos"), ("v2_cos", "v2", "cos"), ("hopt1_cos", "hopt1", "cos"),
           ("uni2_l1", "uni2", "l1"), ("v2_l1", "v2", "l1"), ("hopt1_l1", "hopt1", "l1")]
TILES_PER_TASK = {"cos": 500_000, "l1": 150_000}


def main() -> int:
    sel = json.loads((ROOT / "canary/canary_selection.json").read_text())
    sentinel = sel["sentinel"]
    s = pd.read_csv(ROOT / "banks/uni2/slides.csv", dtype={"slide_id": str})
    cmds, plan = [], []
    for cfg, bank, metric in CONFIGS:
        ref = ROOT / f"canary/hits/{cfg}/top_10_hit_slides_slide_{sentinel}.pkl"
        if not ref.exists():
            raise SystemExit(f"missing sentinel reference {ref}")
        groups, cur, n = [], [], 0
        for sid, k in zip(s["slide_id"], s["n_rows"]):
            if cur and n + k > TILES_PER_TASK[metric]:
                groups.append((cur, n))
                cur, n = [], 0
            cur.append(sid)
            n += int(k)
        groups.append((cur, n))
        for i, (ids, n) in enumerate(groups):
            task = f"{cfg}_{i:03d}"
            cmds.append(f"{PY} {HARNESS} --bank {ROOT}/banks/{bank} --metric {metric} --config {cfg} "
                        f"--slides {','.join(ids)} --out-root {ROOT} --task {task} "
                        f"--sentinel {sentinel} --sentinel-ref {ref}")
            plan.append(dict(task=task, config=cfg, n_slides=len(ids), n_tiles=n, first=ids[0], last=ids[-1]))
    (ROOT / "pool_cmds/production.txt").write_text("\n".join(cmds) + "\n")
    pd.DataFrame(plan).to_csv(ROOT / "pool_cmds/production_plan.csv", index=False)
    p = pd.DataFrame(plan)
    print(p.groupby("config").agg(tasks=("task", "size"), slides=("n_slides", "sum"), tiles=("n_tiles", "sum"),
                                  max_tiles=("n_tiles", "max")))
    print("total tasks", len(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
