"""R2.2-ALL Stage A input fingerprinting (one shard of target slides).

SHA-256 of every per-slide input file read by stage_a_slides.py; WSIs (too large
to hash routinely) get a size/mtime/realpath fingerprint. Writes
OUT/bindings/<sid>.json atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
H = Path("/PROSPER_WORKSPACE")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fp(p, hash_it=True):
    p = Path(p)
    if not p.exists():
        return None
    st = os.stat(p)
    d = {"path": str(p), "size": st.st_size, "mtime": st.st_mtime}
    if hash_it:
        d["sha256"] = sha256_file(p)
    else:
        d["realpath"] = os.path.realpath(p)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides-file", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out) / "bindings"
    out.mkdir(parents=True, exist_ok=True)
    full_bank = Path(os.environ["PROSPER_OUT"]) / "features/uni2_all_patches"
    sids = [s.strip() for s in open(a.slides_file) if s.strip()][a.lo:a.hi]
    for sid in sids:
        rec = {
            "slide": sid,
            "wsi": fp(H / f"data/comet_final_HPC/{sid}.svs", hash_it=False),
            "mosaic_dict": fp(H / f"comet-new-scans/data/mosaic_final_final/{sid}.pkl"),
            "all_patches_csv": fp(H / f"comet-annotation/output/all_patches/{sid}.csv"),
            "mosaic_bank": fp(H / f"comet-feature-gen/comet-uni2/data/tmp_final/{sid}.pkl"),
            "full_bank": fp(full_bank / f"{sid}.pkl"),
        }
        tmp = out / f".{sid}.json.tmp.{os.getpid()}"
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, out / f"{sid}.json")
        print(f"[ok] {sid}", flush=True)
    print("[shard-done]", flush=True)


if __name__ == "__main__":
    main()
