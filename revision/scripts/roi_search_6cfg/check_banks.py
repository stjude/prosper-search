#!/usr/bin/env python3
"""R3.5-ROI6 bank gate: all packed banks hold the same 2,921 slides with identical
per-slide row counts and tile coordinates (so hit_patch_id means the same tile in
every configuration), and each bank's recorded hashes match its files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_bank import sha256_file  # noqa: E402


def main(root: Path, names) -> int:
    out, ref = {}, None
    for n in names:
        d = root / n
        meta = json.loads((d / "bank.json").read_text())
        s = pd.read_csv(d / "slides.csv", dtype={"slide_id": str})
        files = dict(features=sha256_file(d / "features.f32") == meta["features_sha256"],
                     coords=sha256_file(d / "coords.i64") == meta["coords_sha256"],
                     slides_csv=sha256_file(d / "slides.csv") == meta["slides_csv_sha256"])
        row = dict(n_slides=meta["n_slides"], n_rows=meta["n_rows"], dim=meta["dim"],
                   src_dtype_counts=meta["src_dtype_counts"], hashes_match=files)
        if ref is None:
            ref = s
        else:
            row["same_slides_rows_coords_as_" + names[0]] = bool(
                s["slide_id"].equals(ref["slide_id"]) and s["n_rows"].equals(ref["n_rows"])
                and s["coords_sha256"].equals(ref["coords_sha256"]))
        out[n] = row
    ok = all(all(r["hashes_match"].values()) for r in out.values()) and \
        all(v for r in out.values() for k, v in r.items() if k.startswith("same_"))
    out["pass"] = ok
    (root / "BANK_CHECK.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), sys.argv[2:]))
