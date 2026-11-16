#!/usr/bin/env python3
"""R2.2-ALL Stage C queue/environment probe: imports, a tiny exact EMD, a tiny bank read
(sha-verified) and, with --gpu, an exact GPU-vs-production Hamming check. Writes one json."""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import stage_c_lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rec = {"ok": False}
    import scipy  # noqa: F401
    import ot
    agg, bob, otm = L.production()
    rng = np.random.default_rng(0)
    x, y = rng.random((50, 8)).astype(np.float32), rng.random((60, 8)).astype(np.float32)
    M = ot.dist(x, y, metric="cosine")
    cost, log = ot.emd2(np.ones(50) / 50, np.ones(60) / 60, M, numItermax=10 ** 6, log=True)
    rec["emd"] = {"cost": float(cost), "result_code": int(log["result_code"])}
    z = L.load_bank("7469")
    rec["bank_read"] = {"slide": "7469", "rows": int(len(z["feats"]))}
    bits = bob.features_to_barcodes(z["feats"])
    rec["hamming_cpu"] = bob.hamming_min_median(bits, bits[::-1])
    if a.gpu:
        import torch
        import stage_c_yottixel as Y
        rec["torch"] = torch.__version__
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["cuda"] = torch.version.cuda
        q = np.unpackbits(np.packbits(bits, axis=1), axis=1, count=L.NBITS)
        lib = [np.packbits(bob.features_to_barcodes(rng.standard_normal((n, L.DIM)).astype(np.float32)),
                           axis=1) for n in (1, 7, 64)]
        fwd, rev, _ = Y.gpu_rows(q, lib, torch)
        want = [bob.hamming_min_median(q, L.unpack_barcodes(p)) for p in lib]
        back = [bob.hamming_min_median(L.unpack_barcodes(p), q) for p in lib]
        rec["gpu_exact"] = bool(list(fwd) == want and list(rev) == back)
    rec["env"] = L.env_record()
    rec["ok"] = bool(rec["emd"]["result_code"] == 1 and (not a.gpu or rec["gpu_exact"]))
    rec["seconds"] = round(time.time() - t0, 2)
    q = os.environ.get("LSB_QUEUE", "local")
    L.dump_json(Path(a.out_dir) / f"{q}_{os.environ.get('LSB_JOBID', os.getpid())}.json", rec)
    print(rec)
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
