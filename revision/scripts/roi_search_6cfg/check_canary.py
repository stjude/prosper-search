#!/usr/bin/env python3
"""R3.5-ROI6 gate G2: canary hits (published bank, emulated published arithmetic and
directory-listing tie order) against the published hit files, row by row."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_search_gpu as G  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PROSPER_OUT", REPO / "prosper_revision_output"))
ROOT = OUT / "roi_search_6cfg_20261111"
PUBLISHED = Path("/PROSPER_WORKSPACE/"
                 "comet-new-scans/output/top_k_hit_slides_final_slides_final")
TOL = 1e-3


def main() -> int:
    sel = json.loads((ROOT / "canary/canary_selection.json").read_text())
    rows, total = [], dict(tiles=0, identical=0, exact=0, unexplained=0)
    for s in sel["canary"]:
        new = pd.read_pickle(ROOT / f"canary/hits/canary_pub/top_10_hit_slides_slide_{s}.pkl")
        pub = pd.read_pickle(PUBLISHED / f"top_10_hit_slides_slide_{s}.pkl")
        c = G.compare_hits(new, pub, TOL)
        rows.append(dict(slide=s, src_dtype=sel["canary_dtypes"][s], **{k: v for k, v in c.items() if k != "unexplained_tiles"},
                         unexplained_examples=c.get("unexplained_tiles", [])[:5]))
        total["tiles"] += c["tiles"] or 0
        total["identical"] += c.get("identical_tiles", 0)
        total["exact"] += c.get("exact_tiles", 0)
        total["unexplained"] += c["n_unexplained"] if c["n_unexplained"] >= 0 else 10 ** 9
    total["identical_fraction"] = total["identical"] / max(total["tiles"], 1)
    total["exact_fraction"] = total["exact"] / max(total["tiles"], 1)
    res = dict(per_slide=rows, total=total, tol=TOL, passed=total["unexplained"] == 0,
               rule="every published row reproduced (hit slide, hit tile, distance) except rows explained by "
                    "rounding at relative tolerance 1e-3 (float16 one-ulp / float32 summation order)")
    (ROOT / "canary/G2_CHECK.json").write_text(json.dumps(res, indent=1, default=str))
    print(pd.DataFrame(rows).drop(columns=["unexplained_examples"]).to_string(index=False))
    print(json.dumps(total))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
