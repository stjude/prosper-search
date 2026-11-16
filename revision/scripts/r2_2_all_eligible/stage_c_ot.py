#!/usr/bin/env python3
"""R2.2-ALL Stage C exact-OT canary.

Production path (imported ``OTSlideEmbeddingGenerator``, UNI2 feature_dim 1536): float32 features,
L2 normalisation with norms floored at 1e-8, ``ot.dist(fa, fb, metric="cosine")`` and uniform
weights summing to one on each slide, exact ``ot.emd2``.

Modes
  c0       one canary query x one block of the 2,921 saved mosaic slides, calling the production
           ``compute_ot_distance`` unchanged (cap 1e6, as published) and recording the POT
           result status from its warnings.
  pair     one all-eligible bank pair (a -> b) in this fresh process: cost formation on the CPU
           (or a GPU-built cost read from --m-from), ``emd2(..., numItermax=cap, log=True)``;
           result_code, warnings, stage times and VmHWM per stage.
  gpu-cost one bank pair: POT's cosine formula in torch float32 (TF32 off) on the GPU, transfer
           to the host, CPU ot.dist timings at 1 and 8 BLAS threads, max |dM|; writes the GPU cost
           to --scratch for a CPU ``pair --m-from`` solve (no GPU is held during a solve).
  row      one query against a list of bank references; each pair solved in a forked child
           (failure isolation and child peak RSS via wait4). Self/same-patient refs are recorded
           as excluded and never solved.

A pair is ACCEPTED only with result_code == 1, a finite cost and no exception. Every other outcome
is a FAILED item with its status; no distance sentinel is ever written for it.
"""
import argparse
import json
import os
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import stage_c_lib as L  # noqa: E402

POT_CODE_TEXT = {0: "infeasible", 1: "optimal", 2: "unbounded", 3: "max_iter_reached"}


def generator(cap):
    _, _, otm = L.production()
    return otm.OTSlideEmbeddingGenerator(metric="cosine", reg=0.0, feature_dim=L.DIM,
                                        verbose=False, max_iter=int(cap))


def norm_feats(gen, feats, sid):
    return gen.process_single_slide(feats, sid)["features"]


def code_from_warnings(ws):
    txt = " | ".join(str(w.message) for w in ws)
    if "numItermax reached" in txt:
        return 3, txt
    if "infeasible" in txt.lower():
        return 0, txt
    if "unbounded" in txt.lower():
        return 2, txt
    return 1, txt


def solve(fa, fb, cap, M=None):
    """CPU cost (unless given) + exact emd2 with log; returns a status dict."""
    import ot
    rec = {"n_a": int(len(fa)), "n_b": int(len(fb)), "cells": int(len(fa)) * int(len(fb)),
           "cap": int(cap)}
    t0 = time.time()
    if M is None:
        M = ot.dist(fa, fb, metric="cosine")
    rec["t_cost"] = time.time() - t0
    rec["vmhwm_after_cost_gb"] = L.vmhwm_gb()
    a = np.ones(len(fa)) / len(fa)
    b = np.ones(len(fb)) / len(fb)
    t1 = time.time()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        cost, log = ot.emd2(a, b, M, numItermax=int(cap), log=True)
    rec["t_solve"] = time.time() - t1
    rec["vmhwm_after_solve_gb"] = L.vmhwm_gb()
    rec["cost"] = float(cost)
    rec["result_code"] = int(log["result_code"])
    rec["warning"] = log.get("warning")
    rec["warnings"] = [str(w.message) for w in ws]
    ok = rec["result_code"] == 1 and np.isfinite(rec["cost"])
    rec["status"] = "accepted" if ok else (
        "failed_" + POT_CODE_TEXT.get(rec["result_code"], "code%d" % rec["result_code"])
        if rec["result_code"] != 1 else "failed_nonfinite")
    return rec


def cmd_c0(a):
    gen = generator(1e6)
    ids = sorted(p.stem for p in L.MOSAIC_BANK.glob("*.pkl"))
    cols = [s for s in ids if s != a.query][a.block::a.nblocks]
    q = norm_feats(gen, L.load_mosaic(a.query).astype(np.float32), a.query)
    out = []
    t0 = time.time()
    for c in cols:
        f = norm_feats(gen, L.load_mosaic(c).astype(np.float32), c)
        t1 = time.time()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            try:
                d = float(gen.compute_ot_distance(q, f))
                code, txt = code_from_warnings(ws)
            except Exception as exc:  # noqa: BLE001
                d, code, txt = float("nan"), -1, repr(exc)
        out.append((c, d, code, txt, time.time() - t1, len(f)))
    L.save_npz(Path(a.out) / f"c0_{a.query}_{a.block:02d}.npz", cols=np.array([o[0] for o in out]),
               d=np.array([o[1] for o in out]), code=np.array([o[2] for o in out]),
               secs=np.array([o[4] for o in out]), n_ref=np.array([o[5] for o in out]))
    L.dump_json(Path(a.out) / f"c0_{a.query}_{a.block:02d}.json",
                {"query": a.query, "block": a.block, "nblocks": a.nblocks, "n_pairs": len(out),
                 "n_query_tiles": int(len(q)), "seconds": time.time() - t0,
                 "codes": {str(k): int(v) for k, v in zip(*np.unique([o[2] for o in out],
                                                                     return_counts=True))},
                 "warnings": sorted({o[3] for o in out if o[3]})[:20],
                 "vmhwm_gb": L.vmhwm_gb(), "env": L.env_record(), "code": L.production_shas()})


def load_pair(gen, sa, sb):
    t0 = time.time()
    fa = norm_feats(gen, L.load_bank(sa)["feats"], sa)
    fb = norm_feats(gen, L.load_bank(sb)["feats"], sb)
    return fa, fb, time.time() - t0


def cmd_pair(a):
    import ot  # noqa: F401
    gen = generator(a.cap)
    rec = {"a": a.a, "b": a.b, "env": L.env_record(), "code": L.production_shas()}
    try:
        fa, fb, t_load = load_pair(gen, a.a, a.b)
        rec["t_load_norm"] = t_load
        rec["vmhwm_after_load_gb"] = L.vmhwm_gb()
        M = None
        if a.m_from:
            t0 = time.time()
            M = np.load(a.m_from)
            rec["t_read_M"] = time.time() - t0
            rec["m_from"] = a.m_from
            if M.shape != (len(fa), len(fb)) or M.dtype != np.float32:
                raise ValueError(f"M {M.shape} {M.dtype}")
        rec.update(solve(fa, fb, a.cap, M))
    except Exception as exc:  # noqa: BLE001
        rec.update(status="failed_exception", error=repr(exc), traceback=traceback.format_exc())
    rec["ru_maxrss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    tag = f"{a.a}_{a.b}_cap{int(a.cap):.0e}".replace("+", "") + ("_gpuM" if a.m_from else "")
    L.dump_json(Path(a.out) / f"pair_{tag}.json", rec)
    print(json.dumps({k: rec.get(k) for k in ("a", "b", "cells", "status", "cost", "result_code",
                                              "t_cost", "t_solve", "vmhwm_after_solve_gb")}))
    # a recorded non-accepted status is an outcome (FAILED item), not a crash to be retried
    return 0


def cmd_gpu_cost(a):
    import ot
    import torch
    from threadpoolctl import threadpool_limits
    import stage_c_yottixel as Y
    Y.configure_torch(torch)
    gen = generator(1e8)
    fa, fb, t_load = load_pair(gen, a.a, a.b)
    rec = {"a": a.a, "b": a.b, "cells": len(fa) * len(fb), "t_load_norm": t_load,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    nslot = int(os.environ.get("OMP_NUM_THREADS", "1"))
    with threadpool_limits(limits=1):
        t0 = time.time()
        Mc1 = ot.dist(fa, fb, metric="cosine")
        rec["t_cost_cpu_1thread"] = time.time() - t0
    with threadpool_limits(limits=nslot):
        t0 = time.time()
        Mc = ot.dist(fa, fb, metric="cosine")
        rec["t_cost_cpu_nthreads"] = time.time() - t0
        rec["cpu_threads"] = nslot
        del Mc
    torch.cuda.synchronize()
    t0 = time.time()
    x1 = torch.from_numpy(fa).cuda()
    x2 = torch.from_numpy(fb).cuda()
    torch.cuda.synchronize()
    rec["t_h2d_feats"] = time.time() - t0
    t1 = time.time()
    n1 = torch.sqrt(torch.einsum("ij,ij->i", x1, x1))
    n2 = torch.sqrt(torch.einsum("ij,ij->i", x2, x2))
    D = x1 @ x2.T
    D.div_(n1[:, None]).div_(n2[None, :]).neg_().add_(1.0)
    torch.cuda.synchronize()
    rec["t_cost_gpu"] = time.time() - t1
    t2 = time.time()
    Mg = D.cpu().numpy()
    rec["t_d2h_M"] = time.time() - t2
    rec["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    del D, x1, x2
    diff = np.abs(Mg - Mc1)
    rec["max_abs_dM"] = float(diff.max())
    rec["n_cells_dM_nonzero"] = int((diff > 0).sum())
    del diff, Mc1
    t3 = time.time()
    path = Path(a.scratch) / f"M_gpu_{a.a}_{a.b}.npy"
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.npy")
    np.save(tmp, Mg)
    os.replace(tmp, path)
    rec["t_write_M"] = time.time() - t3
    rec["m_path"] = str(path)
    rec.update(vmhwm_gb=L.vmhwm_gb(), env=L.env_record(), code=L.production_shas())
    L.dump_json(Path(a.out) / f"gpucost_{a.a}_{a.b}.json", rec)
    print(json.dumps({k: rec[k] for k in rec if k not in ("env", "code")}))
    return 0


def _child(fq, fr, cap, wfd):
    try:
        rec = solve(fq, fr, cap)
    except BaseException as exc:  # noqa: BLE001
        rec = {"status": "failed_exception", "error": repr(exc)}
    os.write(wfd, json.dumps(rec).encode())
    os.close(wfd)
    os._exit(0)


def cmd_row(a):
    gen = generator(a.cap)
    meta = L.meta()
    refs = [r for r in Path(a.refs).read_text().split() if r]
    q = norm_feats(gen, L.load_bank(a.query)["feats"], a.query)
    outp = Path(a.out) / f"row_{a.query}_{a.tag}.jsonl"
    done = set()
    if outp.exists():
        for ln in outp.read_text().splitlines():
            r = json.loads(ln)
            if r.get("status") == "accepted" or r.get("status", "").startswith("excluded"):
                done.add(r["ref"])
    t0 = time.time()
    with open(outp, "a") as fh:
        for r in refs:
            if r in done:
                continue
            rec = {"query": a.query, "ref": r, "cap": int(a.cap)}
            if r == a.query or meta[r][0] == meta[a.query][0]:
                rec["status"] = "excluded_self" if r == a.query else "excluded_same_patient"
            else:
                t1 = time.time()
                fr = norm_feats(gen, L.load_bank(r)["feats"], r)
                rec["t_load_norm"] = time.time() - t1
                rss_fork = L.vmhwm_gb()
                rfd, wfd = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(rfd)
                    _child(q, fr, a.cap, wfd)
                os.close(wfd)
                chunks = []
                while True:
                    b = os.read(rfd, 1 << 16)
                    if not b:
                        break
                    chunks.append(b)
                os.close(rfd)
                _, status, ru = os.wait4(pid, 0)
                try:
                    rec.update(json.loads(b"".join(chunks).decode()))
                except ValueError:
                    rec["status"] = f"failed_child_status_{status}"
                rec["child_maxrss_gb"] = ru.ru_maxrss / 1024 ** 2
                rec["parent_vmhwm_at_fork_gb"] = rss_fork
                rec["child_cpu_s"] = ru.ru_utime + ru.ru_stime
                rec.pop("vmhwm_after_cost_gb", None)
                rec.pop("vmhwm_after_solve_gb", None)
                del fr
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    L.dump_json(Path(a.out) / f"row_{a.query}_{a.tag}.json",
                {"query": a.query, "tag": a.tag, "n_refs": len(refs), "seconds": time.time() - t0,
                 "vmhwm_gb": L.vmhwm_gb(), "env": L.env_record(), "code": L.production_shas()})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["c0", "pair", "gpu-cost", "row"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--query")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--nblocks", type=int, default=1)
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--cap", type=float, default=1e8)
    ap.add_argument("--m-from")
    ap.add_argument("--scratch", default=str(L.STAGE_C / "scratch"))
    ap.add_argument("--refs")
    ap.add_argument("--tag", default="all")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.scratch).mkdir(parents=True, exist_ok=True)
    return {"c0": cmd_c0, "pair": cmd_pair, "gpu-cost": cmd_gpu_cost, "row": cmd_row}[a.mode](a) or 0


if __name__ == "__main__":
    sys.exit(main())
