#!/usr/bin/env python3
"""R3.5-ROI6 ROI search on GPU ([analysis plan], pre-registered 2026-11-11).

Same algorithm and output format as the published
``comet-new-scans/src/update_hits_tables/compute_top_k_hits_lsf.py``: for every tile of a
query slide, the distance to each other library slide is the minimum over that slide's
tiles, the hit tile is the first argmin, and the 10 nearest slides are written as 10 rows
per query tile in ascending distance (``top_10_hit_slides_slide_<id>.pkl``; columns
query_slide_id, query_patch_id, hit_slide_id, hit_patch_id, similarity = distance).
``roi_eval`` groups these rows in consecutive blocks of 10, so the row layout is part of
the specification.

``--arith pub`` emulates the published mixed-precision NumPy arithmetic (float16
difference, float32 sum rounded to float16 when both slides are float16; float32
otherwise) and ``--tie listdir:FILE`` the published directory-listing tie order; both exist
only for the fidelity canary against the published hit files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TOP_K = 10
INF = float("inf")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Bank:
    def __init__(self, d: Path):
        if not (d / "COMPLETE").exists():
            raise SystemExit(f"{d}: bank not COMPLETE")
        self.dir = d
        self.meta = json.loads((d / "bank.json").read_text())
        s = pd.read_csv(d / "slides.csv", dtype={"slide_id": str})
        self.ids = s["slide_id"].tolist()
        self.index = {x: i for i, x in enumerate(self.ids)}
        self.start = s["row_start"].to_numpy(np.int64)
        self.n = s["n_rows"].to_numpy(np.int64)
        self.src16 = (s["src_dtype"] == "float16").to_numpy()
        self.dim = int(self.meta["dim"])
        self.mm = np.memmap(d / "features.f32", dtype=np.float32, mode="r",
                            shape=(int(self.meta["n_rows"]), self.dim))

    def rows(self, i: int) -> np.ndarray:
        return np.asarray(self.mm[self.start[i]:self.start[i] + self.n[i]])


class Library:
    """Bank resident on the device in slide-aligned chunks (a slide never spans two chunks,
    so each slide's minimum and first argmin are computed within one chunk)."""

    def __init__(self, bank: Bank, device, metric: str, chunk_rows: int, keep_half: bool):
        self.bank, self.metric, self.device = bank, metric, device
        self.chunks = []
        S, i = len(bank.ids), 0
        fh = open(bank.dir / "features.f32", "rb")  # plain reads: mapped pages would count against the job memory limit
        while i < S:
            j, rows = i, 0
            while j < S and (j == i or rows + bank.n[j] <= chunk_rows):
                rows += int(bank.n[j])
                j += 1
            r0 = int(bank.start[i])
            fh.seek(r0 * bank.dim * 4)
            X = np.fromfile(fh, dtype=np.float32, count=rows * bank.dim).reshape(rows, bank.dim)
            X = torch.from_numpy(X).to(device)
            if metric == "cos":
                X = torch.nn.functional.normalize(X, dim=1)
            counts = torch.from_numpy(bank.n[i:j])
            seg = torch.repeat_interleave(torch.arange(j - i), counts).to(device)
            pos = torch.cat([torch.arange(int(c), dtype=torch.int32) for c in counts]).to(device)
            ch = dict(X=X, seg=seg, pos=pos, s0=i, ns=j - i, r0=r0)
            if keep_half:
                m16 = torch.from_numpy(np.repeat(bank.src16[i:j], bank.n[i:j])).to(device)
                ch["cols16"] = torch.nonzero(m16).squeeze(1)
                ch["X16"] = X[ch["cols16"]].half()
            self.chunks.append(ch)
            i = j
        fh.close()

    def query(self, i: int) -> torch.Tensor:
        """Query rows as stored in the resident library (normalized for cosine)."""
        b = self.bank
        r0 = int(b.start[i])
        for ch in self.chunks:
            if ch["s0"] <= i < ch["s0"] + ch["ns"]:
                a = r0 - ch["r0"]
                return ch["X"][a:a + int(b.n[i])]
        raise KeyError(i)


def _dist(Q, ch, metric, pub_q16):
    X = ch["X"]
    if metric == "l1":
        D = torch.cdist(Q, X, p=1)
    else:
        D = 1.0 - Q @ X.T
    if pub_q16 and ch["cols16"].numel():
        Qh = Q.half()
        cols, X16 = ch["cols16"], ch["X16"]
        step = max(1, (1 << 28) // max(1, Q.shape[0] * Q.shape[1]))
        for a in range(0, cols.numel(), step):
            d16 = (Qh[:, None, :] - X16[None, a:a + step, :]).abs().float().sum(-1).half().float()
            D[:, cols[a:a + step]] = d16
    return D


def search_tiles(Q, lib: Library, q_index: int, order: torch.Tensor, pub_q16: bool, k: int = TOP_K):
    """Top-k slides for each row of Q. Returns (slide_idx, hit_tile, distance), each (B, k)."""
    B, S = Q.shape[0], len(lib.bank.ids)
    best = torch.full((B, S), INF, device=Q.device, dtype=torch.float32)
    arg = torch.full((B, S), -1, device=Q.device, dtype=torch.int32)
    big = torch.iinfo(torch.int32).max
    for ch in lib.chunks:
        D = _dist(Q, ch, lib.metric, pub_q16)
        seg = ch["seg"].expand(B, -1)
        mins = torch.full((B, ch["ns"]), INF, device=Q.device, dtype=torch.float32)
        mins.scatter_reduce_(1, seg, D, reduce="amin", include_self=True)
        cand = torch.where(D == mins.gather(1, seg), ch["pos"].expand(B, -1), big)
        am = torch.full((B, ch["ns"]), big, device=Q.device, dtype=torch.int32)
        am.scatter_reduce_(1, seg, cand, reduce="amin", include_self=True)
        best[:, ch["s0"]:ch["s0"] + ch["ns"]] = mins
        arg[:, ch["s0"]:ch["s0"] + ch["ns"]] = am
        del D, cand
    best[:, q_index] = INF
    vals, idx = torch.sort(best[:, order], dim=1, stable=True)
    sl = order[idx[:, :k]]
    return sl, arg.gather(1, sl), vals[:, :k]


def search_slide(lib: Library, sid: str, order: torch.Tensor, arith: str, q_batch: int) -> pd.DataFrame:
    b = lib.bank
    i = b.index[sid]
    Qall = lib.query(i)
    pub_q16 = arith == "pub" and bool(b.src16[i])
    out_s, out_t, out_d = [], [], []
    for a in range(0, Qall.shape[0], q_batch):
        s, t, d = search_tiles(Qall[a:a + q_batch], lib, i, order, pub_q16)
        out_s.append(s.cpu().numpy())
        out_t.append(t.cpu().numpy())
        out_d.append(d.cpu().numpy())
    s, t, d = np.concatenate(out_s), np.concatenate(out_t), np.concatenate(out_d)
    n = s.shape[0]
    ids = np.asarray(b.ids, dtype=object)
    return pd.DataFrame({
        "query_slide_id": np.full(n * TOP_K, sid, dtype=object),
        "query_patch_id": np.repeat(np.arange(n, dtype=np.int64), TOP_K),
        "hit_slide_id": ids[s.ravel()],
        "hit_patch_id": t.ravel().astype(np.int64),
        "similarity": d.ravel().astype(np.float32),
    })


def compare_hits(new: pd.DataFrame, ref: pd.DataFrame, tol: float) -> dict:
    """Per-query-tile comparison of two hit tables with the same layout.

    A tile whose top-10 list differs is *explained* only if the two lists are compatible
    with rounding at relative tolerance ``tol``: position-wise distances agree, every slide
    present in only one list lies at the rank-10 boundary, and every shared slide has the
    same distance. Anything else is an unexplained (real) difference."""
    if len(new) != len(ref):
        return dict(tiles=None, identical_fraction=0.0, n_unexplained=-1, reason=f"rows {len(new)} vs {len(ref)}")
    n = len(new) // TOP_K
    if not ((new["query_patch_id"].to_numpy() == ref["query_patch_id"].to_numpy()).all()
            and (new["query_slide_id"].astype(str).to_numpy() == ref["query_slide_id"].astype(str).to_numpy()).all()):
        return dict(tiles=n, identical_fraction=0.0, n_unexplained=-1, reason="query columns differ")
    hs_n = new["hit_slide_id"].astype(str).to_numpy().reshape(n, TOP_K)
    hs_r = ref["hit_slide_id"].astype(str).to_numpy().reshape(n, TOP_K)
    ht_n = new["hit_patch_id"].to_numpy().reshape(n, TOP_K)
    ht_r = ref["hit_patch_id"].to_numpy().reshape(n, TOP_K)
    d_n = new["similarity"].to_numpy(np.float64).reshape(n, TOP_K)
    d_r = ref["similarity"].to_numpy(np.float64).reshape(n, TOP_K)
    same = (hs_n == hs_r).all(1) & (ht_n == ht_r).all(1)
    exact = same & (d_n == d_r).all(1)

    def rel(x, y):
        return np.abs(x - y) / np.maximum(np.abs(y), 1e-12)

    bad, kinds = [], {"slide_order": 0, "hit_tile_only": 0}
    for r in np.where(~same)[0]:
        ok = bool((rel(d_n[r], d_r[r]) <= tol).all())
        a, b = dict(zip(hs_n[r], d_n[r])), dict(zip(hs_r[r], d_r[r]))
        edge = d_r[r][-1] * (1 - tol)
        ok &= all(a[s] >= edge for s in set(a) - set(b)) and all(b[s] >= edge for s in set(b) - set(a))
        ok &= all(rel(a[s], b[s]) <= tol for s in set(a) & set(b))
        kinds["hit_tile_only" if (hs_n[r] == hs_r[r]).all() else "slide_order"] += 1
        if not ok:
            bad.append(int(r))
    rd = rel(d_n[same], d_r[same]) if same.any() else np.zeros(1)
    return dict(tiles=n, identical_tiles=int(same.sum()), identical_fraction=float(same.mean()),
                exact_tiles=int(exact.sum()), max_rel_distance_diff_identical=float(rd.max()),
                differing_kinds=kinds, unexplained_tiles=bad[:50], n_unexplained=len(bad), tol=tol)


def gpu_identity() -> dict:
    d = dict(host=socket.gethostname(), lsb_jobid=os.environ.get("LSB_JOBID"),
             lsb_queue=os.environ.get("LSB_QUEUE"), torch=torch.__version__, cuda=torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.update(gpu=torch.cuda.get_device_name(0), gpu_capability=f"{p.major}.{p.minor}",
                 gpu_mem_gb=round(p.total_memory / 2 ** 30, 1))
    else:
        d.update(gpu=None)
    return d


def tie_order(bank: Bank, spec: str, device) -> torch.Tensor:
    if spec == "numeric":
        return torch.arange(len(bank.ids), device=device)
    if spec.startswith("listdir:"):
        names = [x.strip() for x in Path(spec[8:]).read_text().splitlines() if x.strip().endswith(".pkl")]
        seq = [bank.index[x[:-4]] for x in names]
        if sorted(seq) != list(range(len(bank.ids))):
            raise SystemExit("listdir order does not cover the bank exactly")
        return torch.tensor(seq, device=device)
    raise SystemExit(f"unknown tie spec {spec}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--metric", required=True, choices=["l1", "cos"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--slides", required=True, help="comma-separated query slide ids")
    ap.add_argument("--task", default="adhoc")
    ap.add_argument("--arith", default="fp32", choices=["fp32", "pub"])
    ap.add_argument("--tie", default="numeric")
    ap.add_argument("--sentinel", default=None, help="slide id rerun first and compared with --sentinel-ref")
    ap.add_argument("--sentinel-ref", default=None, type=Path)
    ap.add_argument("--sentinel-min-identical", type=float, default=0.995)
    ap.add_argument("--tol", type=float, default=1e-5, help="relative distance tolerance for the sentinel check")
    ap.add_argument("--q-batch", type=int, default=512)
    ap.add_argument("--chunk-rows", type=int, default=131072)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    t0 = time.time()
    hits_dir = a.out_root / "hits" / a.config
    rec_dir = a.out_root / "records" / a.config
    hits_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    bank = Bank(a.bank)
    ident = gpu_identity()
    base = dict(config=a.config, task=a.task, metric=a.metric, arith=a.arith, tie=a.tie, bank=str(a.bank),
                bank_features_sha256=bank.meta["features_sha256"], harness=str(Path(__file__).resolve()),
                harness_sha256=sha256_file(Path(__file__).resolve()), q_batch=a.q_batch,
                chunk_rows=a.chunk_rows, **ident)
    slides = [s for s in a.slides.split(",") if s]
    todo = []
    for s in slides:
        side, pkl = rec_dir / f"{s}.json", hits_dir / f"top_10_hit_slides_slide_{s}.pkl"
        if side.exists() and pkl.exists() and json.loads(side.read_text()).get("sha256") == sha256_file(pkl):
            continue
        todo.append(s)
    task_rec = dict(base, slides=slides, todo=len(todo), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(json.dumps({k: task_rec[k] for k in ("config", "task", "gpu", "host", "lsb_queue", "todo")}), flush=True)
    if not todo and not a.sentinel:
        return 0

    lib = Library(bank, a.device, a.metric, a.chunk_rows, keep_half=(a.arith == "pub"))
    order = tie_order(bank, a.tie, a.device)
    task_rec["library_load_seconds"] = round(time.time() - t0, 1)
    print(f"library {len(bank.ids)} slides, {len(lib.chunks)} chunks, {task_rec['library_load_seconds']}s", flush=True)

    if a.sentinel:
        ts = time.time()
        df = search_slide(lib, a.sentinel, order, a.arith, a.q_batch)
        ref = pd.read_pickle(a.sentinel_ref)
        cmp_ = compare_hits(df, ref, a.tol)
        cmp_["seconds"] = round(time.time() - ts, 1)
        cmp_["tiles_per_second"] = round(len(df) / TOP_K / max(cmp_["seconds"], 1e-9), 2)
        cmp_["pass"] = bool(cmp_.get("identical_fraction", 0) >= a.sentinel_min_identical and cmp_.get("n_unexplained", 1) == 0)
        task_rec["sentinel"] = dict(slide=a.sentinel, ref=str(a.sentinel_ref), ref_sha256=sha256_file(a.sentinel_ref), **cmp_)
        print("sentinel", json.dumps(task_rec["sentinel"]), flush=True)
        if not cmp_["pass"]:
            task_rec["status"] = "SENTINEL_FAIL"
            (rec_dir / f"task_{a.task}_{ident.get('lsb_jobid')}.json").write_text(json.dumps(task_rec, indent=1))
            return 3

    done = []
    for s in todo:
        ts = time.time()
        df = search_slide(lib, s, order, a.arith, a.q_batch)
        pkl = hits_dir / f"top_10_hit_slides_slide_{s}.pkl"
        tmp = pkl.with_suffix(".pkl.tmp")
        df.to_pickle(tmp)
        os.replace(tmp, pkl)
        sec = round(time.time() - ts, 2)
        side = dict(base, slide=s, tiles=len(df) // TOP_K, rows=len(df), seconds=sec, sha256=sha256_file(pkl),
                    written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        (rec_dir / f"{s}.json").write_text(json.dumps(side, indent=1))
        done.append(dict(slide=s, tiles=side["tiles"], seconds=sec))
        print(f"{s} {side['tiles']} tiles {sec}s ({side['tiles'] / max(sec, 1e-9):.1f}/s)", flush=True)
    task_rec.update(status="COMPLETE", done=done, seconds=round(time.time() - t0, 1),
                    finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (rec_dir / f"task_{a.task}_{ident.get('lsb_jobid')}.json").write_text(json.dumps(task_rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
