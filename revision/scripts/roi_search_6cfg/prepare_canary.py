#!/usr/bin/env python3
"""R3.5-ROI6 canary set-up: fixed canary/sentinel slides (seeded rule, chosen before any GPU
output), the published bank's directory-listing order (the published tie order) and the
canary GPU task list."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
ROOT = OUT / "roi_search_6cfg_20261111"
PUB_BANK = Path("/PROSPER_WORKSPACE/"
                "comet-new-scans/data/raw_feature_virchow_final_slides_combined")
PY = "/HOME/miniconda3/bin/python3"
HARNESS = REPO / "scripts/roi_search_6cfg/roi_search_gpu.py"
CONFIGS = {"v2_l1": ("v2", "l1"), "v2_cos": ("v2", "cos"), "uni2_l1": ("uni2", "l1"),
           "uni2_cos": ("uni2", "cos"), "hopt1_l1": ("hopt1", "l1"), "hopt1_cos": ("hopt1", "cos")}


def main() -> int:
    can = ROOT / "canary"
    can.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ROOT / "banks/v2pub/slides.csv", dtype={"slide_id": str})
    rng = np.random.default_rng(20261111)
    mid = s[(s["n_rows"] >= 200) & (s["n_rows"] <= 1500)]
    p16 = sorted(rng.choice(mid[mid["src_dtype"] == "float16"]["slide_id"].to_numpy(), 6, replace=False), key=int)
    f32 = mid[mid["src_dtype"] == "float32"]["slide_id"].to_numpy()
    p32 = sorted(rng.choice(f32, min(2, len(f32)), replace=False), key=int) if len(f32) else []
    canary = p16 + p32
    n = dict(zip(s["slide_id"], s["n_rows"]))
    sentinel = min(canary, key=lambda x: (abs(n[x] - 800), int(x)))
    names = os.listdir(PUB_BANK)
    (can / "listdir_v2pub.txt").write_text("\n".join(names) + "\n")
    rec = dict(canary=canary, canary_tiles={x: int(n[x]) for x in canary}, canary_dtypes={x: str(
        s.set_index("slide_id").at[x, "src_dtype"]) for x in canary}, sentinel=sentinel, sentinel_tiles=int(n[sentinel]),
        rule="seed 20261111; 6 float16 + 2 float32 slides with 200-1500 tiles; sentinel = canary slide closest to 800 tiles",
        listdir_entries=len(names), listdir_captured=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        published_bank_dir_mtime=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(PUB_BANK.stat().st_mtime)))
    (can / "canary_selection.json").write_text(json.dumps(rec, indent=1))
    cmds = [f"{PY} {HARNESS} --bank {ROOT}/banks/v2pub --metric l1 --config canary_pub --arith pub "
            f"--tie listdir:{can}/listdir_v2pub.txt --slides {','.join(canary)} --q-batch 32 --out-root {can} --task canary_pub"]
    for cfg, (bank, metric) in CONFIGS.items():
        cmds.append(f"{PY} {HARNESS} --bank {ROOT}/banks/{bank} --metric {metric} --config {cfg} "
                    f"--slides {sentinel} --out-root {can} --task sentinel_ref")
    (ROOT / "pool_cmds/canary.txt").write_text("\n".join(cmds) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
