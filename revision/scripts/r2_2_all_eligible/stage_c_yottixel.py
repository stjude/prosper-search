#!/usr/bin/env python3
"""R2.2-ALL Stage C Yottixel (BoB) canary: production CPU path and an exact GPU kernel.

Production semantics (imported from prosper-search ``bob_embedding``): a tile's barcode is
``np.diff(feats, axis=1) < 0`` (1,535 bits for UNI2); the directed slide distance d(q->r) is
``np.median`` over q's tiles of the minimum Hamming count to any tile of r.

GPU kernel (``gpu_rows``): float32 with TF32/reduced-precision reductions disabled. Inputs are
0/1 and every dot product, popcount and Hamming count is an integer <= 1,535 < 2**24, so the
float32 identity |a| + |b| - 2 a.b is exact whatever the summation order. One pass over each
reference slide, in bounded blocks of reference rows, yields the row minima (q->r) and the column
minima (r->q) independently; both medians are then taken on the host with ``np.median`` on the
float32 minima, exactly as ``hamming_min_median`` does (NumPy's even-count median).

Modes
  c0-cpu   one query, production ``hamming_min_median`` against the saved 15 % mosaic bank
  c0-gpu   all canary queries, GPU kernel against the saved 15 % mosaic bank (both directions)
  rows-gpu canary queries against the 2,778 all-eligible library (variant A = admitted bank,
           variant B = code-0 rows replaced by their fp16-autocast fully-tissue rows)
  cpu-ref  production ``hamming_min_median`` on bank features for full rows (--all-columns) or
           for 30 columns per query (GPU top-10 + 20 seeded size-stratified columns)
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import stage_c_lib as L  # noqa: E402

BUDGET = 1 << 28  # float32 elements per distance block (1 GiB)


def configure_torch(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")


def gpu_rows(q_bits, lib_packed, torch, budget=BUDGET, dev="cuda"):
    """Directed medians q->r (fwd) and r->q (rev) for every packed reference slide."""
    configure_torch(torch)
    Q = torch.from_numpy(np.ascontiguousarray(q_bits, dtype=np.float32)).to(dev)
    sq = Q.sum(1)
    shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)
    fwd, rev, secs = [], [], []
    chunk = max(1, budget // max(1, Q.shape[0]))
    for p in lib_packed:
        t0 = time.time()
        P = p if torch.is_tensor(p) else torch.from_numpy(np.ascontiguousarray(p)).to(dev)
        R = ((P.unsqueeze(2) >> shifts) & 1).reshape(P.shape[0], -1)[:, :L.NBITS].float()
        sr = R.sum(1)
        rmin = torch.full((Q.shape[0],), float("inf"), device=dev)
        cmin = torch.empty(R.shape[0], device=dev)
        for s in range(0, R.shape[0], chunk):
            d = sq[:, None] + sr[None, s:s + chunk] - 2.0 * (Q @ R[s:s + chunk].T)
            rmin = torch.minimum(rmin, d.amin(1))
            cmin[s:s + chunk] = d.amin(0)
        fwd.append(float(np.median(rmin.cpu().numpy())))
        rev.append(float(np.median(cmin.cpu().numpy())))
        secs.append(time.time() - t0)
    return fwd, rev, secs


def mosaic_barcodes(ids, bob):
    return {s: bob.features_to_barcodes(L.load_mosaic(s)) for s in ids}


def mosaic_ids():
    return sorted(p.stem for p in L.MOSAIC_BANK.glob("*.pkl"))


def cmd_c0_cpu(a, bob):
    ids = mosaic_ids()
    t0 = time.time()
    bc = mosaic_barcodes(ids, bob)
    t_load = time.time() - t0
    q = a.query
    cols, fwd = [], []
    t1 = time.time()
    for r in ids:
        if r == q:
            continue
        cols.append(r)
        fwd.append(bob.hamming_min_median(bc[q], bc[r]))
    L.save_npz(Path(a.out) / f"c0_cpu_{q}.npz", cols=np.array(cols), fwd=np.array(fwd))
    L.dump_json(Path(a.out) / f"c0_cpu_{q}.json",
                {"query": q, "n_cols": len(cols), "load_s": t_load, "row_s": time.time() - t1,
                 "vmhwm_gb": L.vmhwm_gb(), "env": L.env_record(), "code": L.production_shas()})


def cmd_c0_gpu(a, bob):
    import torch
    ids = mosaic_ids()
    t0 = time.time()
    bc = mosaic_barcodes(ids, bob)
    lib = [torch.from_numpy(np.packbits(bc[s], axis=1)).cuda() for s in ids]
    t_load = time.time() - t0
    rec = {"gpu": torch.cuda.get_device_name(0), "load_s": t_load, "queries": {}}
    for q in [x["slide"] for x in L.selection()["queries"]]:
        t1 = time.time()
        fwd, rev, _ = gpu_rows(bc[q], lib, torch)
        L.save_npz(Path(a.out) / f"c0_gpu_{q}.npz", cols=np.array(ids), fwd=np.array(fwd),
                   rev=np.array(rev))
        rec["queries"][q] = {"row_s": time.time() - t1}
    rec.update(vmhwm_gb=L.vmhwm_gb(), env=L.env_record(), code=L.production_shas(),
               torch=torch.__version__, cuda=torch.version.cuda)
    L.dump_json(Path(a.out) / "c0_gpu.json", rec)


def load_library_packed(variant, slides):
    prep = L.STAGE_C / "prep"
    out = {}
    for s in slides:
        p = np.load(prep / "barcodes" / f"{s}.npy")
        if variant == "B":
            fb = prep / "barcodesB" / f"{s}.npz"
            with np.load(fb) as z:
                if len(z["rows"]):
                    p = p.copy()
                    p[z["rows"]] = z["packedB"]
        out[s] = p
    return out


def cmd_rows_gpu(a, bob):
    import torch
    slides = L.bank_slides()
    t0 = time.time()
    packed = load_library_packed(a.variant, slides)
    lib = [torch.from_numpy(packed[s]).cuda() for s in slides]
    t_load = time.time() - t0
    queries = a.queries.split(",") if a.queries else [x["slide"] for x in L.selection()["queries"]]
    rec = {"variant": a.variant, "gpu": torch.cuda.get_device_name(0), "load_s": t_load,
           "queries": {}}
    for q in queries:
        t1 = time.time()
        qb = L.unpack_barcodes(packed[q])
        fwd, rev, secs = gpu_rows(qb, lib, torch)
        L.save_npz(Path(a.out) / f"rows_{a.variant}_{q}.npz", cols=np.array(slides),
                   fwd=np.array(fwd), rev=np.array(rev), secs=np.array(secs))
        rec["queries"][q] = {"n_tiles": int(len(qb)), "row_s": time.time() - t1}
        print(f"[yottixel] {a.variant} {q} n={len(qb)} {time.time() - t1:.1f}s", flush=True)
    rec.update(vmhwm_gb=L.vmhwm_gb(), env=L.env_record(), code=L.production_shas(),
               torch=torch.__version__, cuda=torch.version.cuda,
               gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)
    L.dump_json(Path(a.out) / f"rows_{a.variant}_{a.tag}.json", rec)


def ref_columns(q, gpu_npz, meta, sizes, seed):
    """GPU top-10 (population, self/same-patient removed) + 20 size-stratified columns."""
    with np.load(gpu_npz) as z:
        cols, fwd = list(z["cols"]), z["fwd"]
    pq = meta[q][0]
    keep = [i for i, c in enumerate(cols) if c != q and meta[c][0] != pq]
    order = sorted(keep, key=lambda i: (fwd[i], i))
    top = [cols[i] for i in order[:10]]
    rng = np.random.default_rng(seed + int(q))
    rest = [cols[i] for i in keep if cols[i] not in top]
    rest.sort(key=lambda c: (sizes[c], int(c)))
    strata = np.array_split(np.array(rest), 5)
    pick = [str(x) for s in strata for x in rng.choice(s, size=4, replace=False)]
    return top + pick


def cmd_cpu_ref(a, bob):
    meta = L.meta()
    slides = L.bank_slides()
    import pandas as pd
    sizes = dict(pd.read_csv(L.STAGE_B / "bank_per_slide.csv", dtype={"slide": str})
                 [["slide", "n_rows"]].itertuples(index=False))
    queries = a.queries.split(",")
    qbits = {q: bob.features_to_barcodes(L.load_bank(q)["feats"]) for q in queries}
    if a.all_columns:
        want = {q: [s for s in slides if s != q] for q in queries}
    else:
        want = {q: ref_columns(q, Path(a.gpu_dir) / f"rows_A_{q}.npz", meta, sizes, a.seed)
                for q in queries}
    need = sorted({c for v in want.values() for c in v}, key=int)
    res = {q: {} for q in queries}
    t0 = time.time()
    prep = L.STAGE_C / "prep" / "barcodes"
    for c in need:
        if c in qbits:
            cb = qbits[c]
        elif a.all_columns:
            # full rows read the prep's production barcodes (round-trip checked at prep); the
            # 30-column check re-derives them from the bank features (independent of prep)
            cb = L.unpack_barcodes(np.load(prep / f"{c}.npy"))
        else:
            cb = bob.features_to_barcodes(L.load_bank(c)["feats"])
        for q in queries:
            if c in want[q]:
                res[q][c] = bob.hamming_min_median(qbits[q], cb)
    for q in queries:
        cols = want[q]
        L.save_npz(Path(a.out) / f"cpuref_{q}.npz", cols=np.array(cols),
                   fwd=np.array([res[q][c] for c in cols]))
    L.dump_json(Path(a.out) / f"cpuref_{a.tag}.json",
                {"queries": queries, "all_columns": a.all_columns, "n_pairs": sum(map(len, want.values())),
                 "seconds": time.time() - t0, "vmhwm_gb": L.vmhwm_gb(), "env": L.env_record(),
                 "code": L.production_shas()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["c0-cpu", "c0-gpu", "rows-gpu", "cpu-ref"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--query")
    ap.add_argument("--queries", default="")
    ap.add_argument("--variant", choices=["A", "B"], default="A")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--all-columns", action="store_true")
    ap.add_argument("--gpu-dir")
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args(argv)
    _, bob, _ = L.production()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"c0-cpu": cmd_c0_cpu, "c0-gpu": cmd_c0_gpu, "rows-gpu": cmd_rows_gpu,
     "cpu-ref": cmd_cpu_ref}[a.mode](a, bob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
