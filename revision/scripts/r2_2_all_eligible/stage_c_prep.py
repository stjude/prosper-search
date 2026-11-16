#!/usr/bin/env python3
"""R2.2-ALL Stage C prep shard: per admitted bank slide, in one read of its npz (sha-verified):

  * SAMPLER vector by the imported production ``aggregate_cli._pool(feats, "sampler")``;
  * packed production barcodes (``np.packbits`` over the 1,535 bits of
    ``bob_embedding.features_to_barcodes``) with an unpack round-trip check;
  * Yottixel sign-sensitivity variant B: every code-0 (fp32 mosaic-bank) row whose coordinate is
    in the fp16-autocast fully-tissue bank gets that row's barcode; per-tile bit-flip counts are
    recorded. Code-1 rows are checked byte-equal to the fully-tissue bank.

Outputs under --out: sampler/<shard>.npz, barcodes/<slide>.npy, barcodesB/<slide>.npz,
records/<shard>.json.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import stage_c_lib as L  # noqa: E402


def one(slide, agg, bob, out):
    rec = {"slide": slide}
    t0 = time.time()
    z = L.load_bank(slide)
    feats, src = z["feats"], z["source"]
    rec["n_rows"] = int(len(feats))
    rec["t_load"] = time.time() - t0
    t1 = time.time()
    vec = agg._pool(feats, "sampler")
    rec["t_sampler"] = time.time() - t1
    rec["sampler_dtype"] = str(vec.dtype)
    t2 = time.time()
    bits = bob.features_to_barcodes(feats)
    packed = np.packbits(bits, axis=1)
    if not np.array_equal(L.unpack_barcodes(packed), bits):
        raise ValueError(f"{slide}: packbits round trip failed")
    rec["t_barcode"] = time.time() - t2
    np.save(out / "barcodes" / f".{slide}.tmp.npy", packed)
    (out / "barcodes" / f".{slide}.tmp.npy").rename(out / "barcodes" / f"{slide}.npy")
    t3 = time.time()
    rows = np.zeros(0, np.int64)
    packedB = np.zeros((0, packed.shape[1]), np.uint8)
    flips = np.zeros(0, np.int32)
    fb = L.load_full_bank(slide)
    code1_equal = None
    if fb is not None:
        ffeat, fxy = fb
        m0 = np.nonzero(src == L.CODE_MOSAIC)[0]
        qpos, fpos = L.partial_match(fxy, z["xy"][m0])
        rows = m0[qpos]
        bB = bob.features_to_barcodes(ffeat[fpos])
        flips = (bB != bits[rows]).sum(1).astype(np.int32)
        packedB = np.packbits(bB, axis=1)
        m1 = np.nonzero(src == L.CODE_FULL)[0]
        q1, f1 = L.partial_match(fxy, z["xy"][m1])
        code1_equal = bool(len(q1) == len(m1) and np.array_equal(feats[m1[q1]], ffeat[f1]))
        if not code1_equal:
            raise ValueError(f"{slide}: code-1 rows differ from the fully-tissue bank")
    L.save_npz(out / "barcodesB" / f"{slide}.npz", rows=rows, packedB=packedB, flips=flips)
    rec.update(t_variantB=time.time() - t3, n_code0=int((src == L.CODE_MOSAIC).sum()),
               n_shared=int(len(rows)), flip_bits_total=int(flips.sum()),
               flip_tiles=int((flips > 0).sum()), flip_max=int(flips.max()) if len(flips) else 0,
               flip_hist=np.bincount(flips, minlength=1).tolist() if len(flips) else [],
               code1_equal_full_bank=code1_equal, t_total=time.time() - t0)
    return vec, rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    for d in ("sampler", "barcodes", "barcodesB", "records"):
        (out / d).mkdir(parents=True, exist_ok=True)
    agg, bob, _ = L.production()
    slides = L.bank_slides()[a.shard::a.nshards]
    vecs, recs = [], []
    for s in slides:
        v, r = one(s, agg, bob, out)
        vecs.append(v)
        recs.append(r)
        print(f"[prep] {s} n={r['n_rows']} {r['t_total']:.1f}s", flush=True)
    L.save_npz(out / "sampler" / f"{a.shard:03d}.npz", ids=np.array(slides), vecs=np.vstack(vecs))
    L.dump_json(out / "records" / f"{a.shard:03d}.json",
                {"shard": a.shard, "nshards": a.nshards, "slides": recs, "vmhwm_gb": L.vmhwm_gb(),
                 "env": L.env_record(), "code": L.production_shas()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
