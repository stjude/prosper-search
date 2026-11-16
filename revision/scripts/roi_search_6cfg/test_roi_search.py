#!/usr/bin/env python3
"""G1 for R3.5-ROI6: the GPU harness (run on CPU here) equals a direct NumPy transcription
of the published ``compute_top_k_hits_lsf.py`` loop on synthetic banks. Run:
``python3 scripts/roi_search_6cfg/test_roi_search.py`` (or pytest)."""
from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack_bank  # noqa: E402
import roi_search_gpu as G  # noqa: E402


def make_src(d: Path, ids, sizes, dim, gen, dtypes):
    d.mkdir(parents=True, exist_ok=True)
    for s, n, dt in zip(ids, sizes, dtypes):
        f = gen(n, dim).astype(dt)
        locs = [{"wsi_loc": [int(i) * 512, 7]} for i in range(n)]
        with open(d / f"{s}.pkl", "wb") as fh:
            pickle.dump((f, locs), fh)


def published_loop(src: Path, qid: str, ref_ids, metric: str) -> pd.DataFrame:
    """The published algorithm, verbatim apart from the metric switch."""
    load = lambda s: pickle.load(open(src / f"{s}.pkl", "rb"))[0]  # noqa: E731
    q = load(qid)
    rows = []
    for qi, qp in enumerate(q):
        sd = {}
        for r in ref_ids:
            R = load(r)
            if metric == "l1":
                dist = np.abs(R - qp).sum(axis=1)
            else:
                Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
                dist = 1.0 - Rn @ (qp / np.linalg.norm(qp))
            sd[r] = (np.min(dist), np.argmin(dist))
        for r, (m, a) in sorted(sd.items(), key=lambda x: x[1][0])[:10]:
            rows.append(dict(query_slide_id=qid, query_patch_id=qi, hit_slide_id=r, hit_patch_id=a, similarity=m))
    return pd.DataFrame(rows)


def run_case(tmp: Path, name, ids, sizes, dim, gen, dtypes, metric, arith, tie_perm, chunk_rows, q_batch, exact):
    src, out = tmp / f"src_{name}", tmp / f"bank_{name}"
    make_src(src, ids, sizes, dim, gen, dtypes)
    assert pack_bank.main(["--name", name, "--src-dir", str(src), "--out-dir", str(out),
                           "--expect-dim", str(dim), "--expect-slides", str(len(ids))]) == 0
    bank = G.Bank(out)
    lib = G.Library(bank, "cpu", metric, chunk_rows, keep_half=(arith == "pub"))
    assert len(lib.chunks) > 2
    if tie_perm is None:
        order_ids, spec = sorted(ids, key=int), "numeric"
    else:
        order_ids = [ids[i] for i in tie_perm]
        f = tmp / f"listdir_{name}.txt"
        f.write_text("\n".join(f"{s}.pkl" for s in order_ids) + "\n")
        spec = f"listdir:{f}"
    order = G.tie_order(bank, spec, "cpu")
    for qid in ids:
        got = G.search_slide(lib, qid, order, arith, q_batch)
        exp = published_loop(src, qid, [s for s in order_ids if s != qid], metric)
        assert len(got) == len(exp) == 10 * sizes[ids.index(qid)]
        assert list(got.columns) == ["query_slide_id", "query_patch_id", "hit_slide_id", "hit_patch_id", "similarity"]
        assert got["hit_slide_id"].tolist() == exp["hit_slide_id"].tolist(), (name, qid)
        assert got["hit_patch_id"].tolist() == exp["hit_patch_id"].astype(np.int64).tolist(), (name, qid)
        assert got["query_patch_id"].tolist() == exp["query_patch_id"].tolist()
        gd, ed = got["similarity"].to_numpy(np.float64), exp["similarity"].to_numpy(np.float64)
        if exact:
            assert np.array_equal(gd, ed), (name, qid, np.abs(gd - ed).max())
        else:
            assert np.allclose(gd, ed, rtol=1e-6, atol=1e-6), (name, qid)
        c = G.compare_hits(got, exp, 1e-6)
        assert c["identical_fraction"] == 1.0 and c["n_unexplained"] == 0
    return bank, lib, order


def test_all():
    rng = np.random.default_rng(7)
    ids = [str(x) for x in (9, 10, 11, 100, 101, 7, 8, 2000, 3, 55, 56, 57)]
    sizes = [4, 11, 1, 6, 3, 9, 2, 5, 7, 12, 3, 4]
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        # (a) L1, small integer features: many exact ties between and within slides
        run_case(tmp, "a", ids, sizes, 6, lambda n, d: rng.integers(0, 3, (n, d)), [np.float32] * 12,
                 "l1", "fp32", None, 8, 3, exact=True)
        # (b) published mixed precision: float16 slides on a coarse grid (sums exact in float32,
        # rounded to float16 at the end), float32 slides, shuffled directory-listing tie order
        grid = lambda n, d: rng.integers(-256, 257, (n, d)) / 64.0  # noqa: E731
        dts = [np.float16 if i % 3 else np.float32 for i in range(12)]
        bank, lib, order = run_case(tmp, "b", ids, sizes, 8, grid, dts, "l1", "pub", rng.permutation(12), 10, 4,
                                    exact=True)
        assert bank.src16.sum() == 8
        # (c) cosine, continuous features
        run_case(tmp, "c", ids, sizes, 16, lambda n, d: rng.normal(size=(n, d)), [np.float32] * 12,
                 "cos", "fp32", None, 9, 5, exact=False)
        # compare_hits flags a real reordering and tolerates a boundary near-tie
        df = G.search_slide(lib, "10", order, "pub", 4)
        d = df["similarity"].to_numpy().reshape(-1, 10)
        t0 = int(np.where(d[:, 0] < d[:, 3])[0][0])
        r = df.index[t0 * 10:(t0 + 1) * 10]
        bad = df.copy()
        bad.loc[r[0], "hit_slide_id"], bad.loc[r[3], "hit_slide_id"] = df.loc[r[3], "hit_slide_id"], df.loc[r[0], "hit_slide_id"]
        assert G.compare_hits(bad, df, 1e-6)["n_unexplained"] == 1
        r = df.index[:10]
        edge = df.copy()
        edge.loc[r[9], "hit_slide_id"] = "999999"
        c = G.compare_hits(edge, df, 1e-6)
        assert c["n_unexplained"] == 0 and c["identical_fraction"] < 1.0
    print("test_roi_search: all checks passed")


if __name__ == "__main__":
    test_all()
