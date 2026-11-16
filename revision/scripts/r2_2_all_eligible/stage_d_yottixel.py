#!/usr/bin/env python3
"""R2.2-ALL Stage D Yottixel production: all 2,778 query rows against the 2,778-slide library.

Kernel: ``stage_c_yottixel.gpu_rows`` imported unchanged from the frozen Stage C code (C0: GPU
forward = production CPU row; C2: GPU rows = production ``hamming_min_median`` on full rows and on
30 columns per canary, exact). Variant A only (the admitted bank barcodes from Stage C prep).

Modes
  plan      chunk the 2,778 queries into tasks by summed query-tile count (LPT bin packing)
  rows      one task: every query of the task; resumable (a valid existing row is skipped);
            fails (rc 3) unless the device is an NVIDIA A100-SXM4-80GB
  assemble  the full directed matrix d[q, r] = fwd(q -> r) with checks (see ``cmd_assemble``)
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import stage_c_lib as L  # noqa: E402
import stage_c_yottixel as Y  # noqa: E402

STAGE_D = L.RUN / "stage_d"
ADMITTED_GPU = "NVIDIA A100-SXM4-80GB"
MS_PER_TILE = 4.85  # measured in C2 (12 GPU tasks, A100-80G): 4.82-4.90 ms per query tile


def sizes():
    import pandas as pd
    t = pd.read_csv(L.STAGE_B / "bank_per_slide.csv", dtype={"slide": str})
    return dict(zip(t["slide"], t["n_rows"].astype(int)))


def cmd_plan(a):
    n = sizes()
    slides = L.bank_slides()
    if sorted(n) != sorted(slides):
        raise ValueError("bank_per_slide.csv does not cover the bank slides")
    total = sum(n[s] for s in slides)
    k = max(1, round(total / a.tiles_per_task))
    bins = [[0, i, []] for i in range(k)]
    for s in sorted(slides, key=lambda s: (-n[s], int(s))):
        b = min(bins, key=lambda x: (x[0], x[1]))
        b[0] += n[s]
        b[2].append(s)
    tasks = [{"task": f"t{b[1]:03d}", "queries": sorted(b[2], key=int), "query_tiles": b[0],
              "projected_s": b[0] * MS_PER_TILE / 1000.0} for b in bins]
    rec = {"n_queries": len(slides), "total_query_tiles": total, "tiles_per_task_target": a.tiles_per_task,
           "n_tasks": k, "ms_per_tile": MS_PER_TILE,
           "projected_gpu_h_compute": total * MS_PER_TILE / 1000.0 / 3600.0,
           "max_task_projected_s": max(t["projected_s"] for t in tasks), "tasks": tasks}
    L.dump_json(Path(a.manifest), rec)
    print(f"[plan] {k} tasks, {total} query tiles, {rec['projected_gpu_h_compute']:.1f} GPU-h compute, "
          f"max task {rec['max_task_projected_s'] / 60:.1f} min", flush=True)


def row_valid(path, slides):
    try:
        with np.load(path) as z:
            return (list(z["cols"]) == slides and len(z["fwd"]) == len(slides)
                    and bool(np.isfinite(z["fwd"]).all()) and bool(np.isfinite(z["rev"]).all())
                    and str(z["gpu"]) == ADMITTED_GPU)
    except Exception:  # noqa: BLE001 - an unreadable partial file is simply recomputed
        return False


def cmd_rows(a):
    import torch
    gpu = torch.cuda.get_device_name(0)
    if gpu != ADMITTED_GPU:
        print(f"[yottixel-d] device {gpu!r} is not admitted ({ADMITTED_GPU}); refusing", flush=True)
        return 3
    man = json.loads(Path(a.manifest).read_text())
    task = next(t for t in man["tasks"] if t["task"] == a.task)
    slides = L.bank_slides()
    out = Path(a.out)
    todo = [q for q in task["queries"] if not row_valid(out / f"{q}.npz", slides)]
    rec = {"task": a.task, "gpu": gpu, "host": socket.gethostname(), "n_queries": len(task["queries"]),
           "skipped_existing": len(task["queries"]) - len(todo), "queries": {}}
    if todo:
        t0 = time.time()
        packed = Y.load_library_packed("A", slides)
        lib = [torch.from_numpy(packed[s]).cuda() for s in slides]
        rec["load_s"] = time.time() - t0
        env = L.env_record()
        for q in todo:
            t1 = time.time()
            qb = L.unpack_barcodes(packed[q])
            fwd, rev, secs = Y.gpu_rows(qb, lib, torch)
            dt = time.time() - t1
            L.save_npz(out / f"{q}.npz", cols=np.array(slides), fwd=np.array(fwd), rev=np.array(rev),
                       secs=np.array(secs), gpu=np.array(gpu), host=np.array(env["host"]),
                       lsf_job=np.array(str(env.get("lsf_job"))), n_tiles=np.array(len(qb)))
            rec["queries"][q] = {"n_tiles": int(len(qb)), "row_s": dt, "tiles_per_s": len(qb) / dt}
            print(f"[yottixel-d] {q} n={len(qb)} {dt:.1f}s", flush=True)
        rec.update(gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 1024 ** 3,
                   torch=torch.__version__, cuda=torch.version.cuda)
    rec.update(vmhwm_gb=L.vmhwm_gb(), env=L.env_record(), code=L.production_shas(),
               script_sha256=L.sha256_file(Path(__file__)), kernel_sha256=L.sha256_file(Path(Y.__file__)),
               lib_sha256=L.sha256_file(Path(L.__file__)))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    L.dump_json(out.parent / "task_records" / f"{a.task}.{stamp}.json", rec)
    bad = [q for q in task["queries"] if not row_valid(out / f"{q}.npz", slides)]
    if bad:
        print(f"[yottixel-d] {len(bad)} rows invalid after the task: {bad[:5]}", flush=True)
        return 2
    return 0


def cmd_assemble(a, bob):
    """Full directed matrix and checks:
    - every bank slide has a valid row (2,778 x 2,778 fwd, finite, admitted GPU model);
    - transpose consistency: fwd(q -> r) from row q equals rev(q -> r) obtained independently from row r
      (the same integer Hamming counts reduced along the other axis of a separately computed block), exact;
    - the 23 canary rows equal the C2 variant-A rows and the C2 production-CPU references (full rows for
      3 queries, 30 columns for every canary), exact;
    - prep barcodes re-derived with production ``features_to_barcodes`` from the admitted bank features
      for a seeded sample of slides are identical to the packed barcodes the kernel read."""
    slides = L.bank_slides()
    out = Path(a.out)
    rows = out.parent / "rows"
    n = len(slides)
    fwd = np.empty((n, n))
    rev = np.empty((n, n))
    prov = {}
    missing = []
    for i, q in enumerate(slides):
        p = rows / f"{q}.npz"
        if not row_valid(p, slides):
            missing.append(q)
            continue
        with np.load(p) as z:
            fwd[i], rev[i] = z["fwd"], z["rev"]
            prov[q] = {"gpu": str(z["gpu"]), "host": str(z["host"]), "lsf_job": str(z["lsf_job"]),
                       "n_tiles": int(z["n_tiles"]), "row_sha256": L.sha256_file(p)}
    checks = {"n_rows_valid": n - len(missing), "missing": missing}
    if missing:
        L.dump_json(out / "yottixel_matrix.json", {"pass": False, "checks": checks})
        print(f"[assemble] {len(missing)} rows missing", flush=True)
        return 2
    checks["finite"] = bool(np.isfinite(fwd).all())
    checks["transpose_consistency_exact"] = bool((fwd == rev.T).all())
    checks["transpose_mismatch_cells"] = int((fwd != rev.T).sum())
    pos = {s: i for i, s in enumerate(slides)}
    c2 = L.STAGE_C / "c2"
    canary = [x["slide"] for x in L.selection()["queries"]]
    gpu_eq, ref_eq, nref = {}, {}, 0
    for q in canary:
        g = c2 / f"rows_A_{q}.npz"
        if g.exists():
            with np.load(g) as z:
                gpu_eq[q] = bool(list(z["cols"]) == slides and (z["fwd"] == fwd[pos[q]]).all())
        for r in (c2 / f"cpuref_{q}.npz", c2 / "ref30" / f"cpuref_{q}.npz"):
            if r.exists():
                with np.load(r) as z:
                    cols = [str(c) for c in z["cols"]]
                    ref_eq[f"{r.parent.name}/{q}"] = bool((fwd[pos[q], [pos[c] for c in cols]] == z["fwd"]).all())
                    nref += len(cols)
    checks.update(canary_gpu_rows_compared=len(gpu_eq), canary_gpu_rows_equal=all(gpu_eq.values()),
                  canary_cpu_refs_compared=len(ref_eq), canary_cpu_ref_cells=nref,
                  canary_cpu_refs_equal=all(ref_eq.values()), canary_detail={"gpu": gpu_eq, "cpu": ref_eq})
    rng = np.random.default_rng(a.seed)
    sample = sorted(rng.choice(slides, size=a.barcode_sample, replace=False).tolist(), key=int)
    bc_ok = {}
    for s in sample:
        packed = np.load(L.STAGE_C / "prep" / "barcodes" / f"{s}.npy")
        bc_ok[s] = bool(np.array_equal(L.unpack_barcodes(packed),
                                       bob.features_to_barcodes(L.load_bank(s)["feats"]).astype(np.uint8)))
    checks.update(barcode_rederive_sample=sample, barcode_rederive_equal=all(bc_ok.values()))
    gpus = sorted({v["gpu"] for v in prov.values()})
    checks["gpu_models"] = gpus
    meta = L.meta()
    pat = np.array([meta[s][0] for s in slides], dtype=object)
    excluded = pat[:, None] == pat[None, :]
    ok = (checks["finite"] and checks["transpose_consistency_exact"] and checks["canary_gpu_rows_equal"]
          and checks["canary_cpu_refs_equal"] and len(gpu_eq) == len(canary)
          and checks["barcode_rederive_equal"] and gpus == [ADMITTED_GPU])
    L.save_npz(out / "yottixel_matrix.npz", ids=np.array(slides), d=fwd, excluded=excluded)
    bsha = {p.name: L.sha256_file(p) for p in sorted((L.STAGE_C / "prep" / "barcodes").glob("*.npy"))}
    L.dump_json(out / "prep_barcodes_SHA256.json", bsha)
    rec = {"arm": "yottixel", "variant": "A", "n": n, "direction": "row q = query, d[q, r] = median over q tiles "
           "of the min Hamming count to r's tiles (production hamming_min_median orientation)",
           "checks": checks, "pass": bool(ok), "row_provenance": prov,
           "matrix_sha256": L.sha256_file(out / "yottixel_matrix.npz"),
           "vmhwm_gb": L.vmhwm_gb(), "env": L.env_record(), "code": L.production_shas(),
           "script_sha256": L.sha256_file(Path(__file__)), "kernel_sha256": L.sha256_file(Path(Y.__file__)),
           "lib_sha256": L.sha256_file(Path(L.__file__))}
    L.dump_json(out / "yottixel_matrix.json", rec)
    print(f"[assemble] pass={ok} " + json.dumps({k: v for k, v in checks.items() if k not in
                                                ("canary_detail", "barcode_rederive_sample", "missing")}), flush=True)
    return 0 if ok else 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "rows", "assemble"])
    ap.add_argument("--manifest", default=str(STAGE_D / "yottixel" / "task_manifest.json"))
    ap.add_argument("--tiles-per-task", type=int, default=250000)
    ap.add_argument("--task")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--barcode-sample", type=int, default=24)
    a = ap.parse_args(argv)
    if a.mode == "plan":
        return cmd_plan(a) or 0
    if a.mode == "rows":
        a.out = a.out or str(STAGE_D / "yottixel" / "rows")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        return cmd_rows(a)
    a.out = a.out or str(STAGE_D / "yottixel" / "matrix")
    _, bob, _ = L.production()
    return cmd_assemble(a, bob)


if __name__ == "__main__":
    sys.exit(main())
