#!/usr/bin/env python3
"""R2.2-ALL Stage C collectors (read-only over Stage C outputs and the published matrices).

c1   SAMPLER 100 % rows: exclusion rule, finiteness, timings, Stage D projection.
c2   Yottixel 100 %: GPU-vs-CPU exactness (full rows, 30 columns), bit-flip statistics over the
     shared tiles, variant A vs B |dd|, top-5 set and hit@5 (tie-bounded) decision changes; GPU
     time projection. hit@5 here is a sensitivity count, not an evaluation result.
c3   exact OT 100 %: pair probes, iteration bracket, orientation consistency, GPU-cost benchmark,
     rows; time/RSS fit vs cells and the Stage D projection.
mp   median-pooling arm (author amendment 2026-11-13 ~21:15 CST): C0 against the published mosaic
     median-pooling matrix (same tolerance and ranking check as SAMPLER) -> c0/c0_median_pooling_summary.json;
     C1 rows (exclusion, finiteness) and the prep cross-checks -> c1/c1_median_pooling_summary.json.
c0   C0 reference reproduction: per arm, per query, max |rel diff| (SAMPLER, OT) or exact equality
     (Yottixel CPU path; GPU kernel vs CPU path; GPU reverse direction vs the published column)
     against the unmasked published matrix, plus the pre-registered top-10 ranking check on the
     2,778 population against the `_no_same_patient` matrix. Writes c0/c0_summary.json + .md.
"""
import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import stage_c_lib as L  # noqa: E402

PUB = {"sampler": "sampler", "yottixel": "yottixel", "ot": "ot_cosine", "median_pooling": "median_pooling"}
TOL = {"sampler": 1e-6, "yottixel": 0.0, "ot": 1e-6, "median_pooling": 1e-6}


def published(arm, masked=False):
    p = L.PUBLISHED / f"combined_similarity_matrix_uni2_{PUB[arm]}{'_no_same_patient' if masked else ''}.pkl"
    with open(p, "rb") as fh:
        return pickle.load(fh), L.sha256_file(p)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def topk(cols, d, k=10):
    order = sorted(range(len(cols)), key=lambda i: (d[i], cols[i]))[:k]
    return [cols[i] for i in order], [float(d[i]) for i in order]


def ranking_check(q, cols, d, pubm, pop, meta, tol):
    """Top-10 on the population, self/same-patient removed, vs the masked published row."""
    pq = meta[q][0]
    keep = [i for i, c in enumerate(cols) if c in pop and c != q and meta[c][0] != pq]
    rc, rd = topk([cols[i] for i in keep], [d[i] for i in keep])
    row = pubm.loc[q]
    pc_all = [c for c in row.index if c in pop and c != q and meta[c][0] != pq]
    pd_all = row[pc_all].to_numpy(float)
    if (pd_all >= 1e6).any():
        raise ValueError(f"{q}: masked published row has sentinel on a kept column")
    pc, pdv = topk(pc_all, pd_all)
    if rc == pc:
        return {"status": "equal", "top10": rc}
    dist_ok = all(abs(x - y) <= tol * max(abs(y), 1e-300) for x, y in zip(rd, pdv))
    diff = set(rc) ^ set(pc)
    pub_d = dict(zip(pc_all, pd_all))
    tie = all(abs(pub_d[c] - pdv[-1]) <= tol * max(abs(pdv[-1]), 1e-300) for c in diff)
    ok = dist_ok and (not diff or tie)
    return {"status": "swap_within_tolerance" if ok else "FAIL", "repro": rc, "published": pc,
            "repro_d": rd, "published_d": pdv}


def c0(out):
    sel = [x["slide"] for x in L.selection()["queries"]]
    meta = L.meta()
    pop = set(L.bank_slides())
    res = {"queries": sel, "tolerances": TOL, "arms": {}}
    base = L.STAGE_C / "c0"
    # SAMPLER
    pub, sha = published("sampler")
    pubm, shm = published("sampler", True)
    with np.load(base / "sampler/sampler_c0.npz") as z:
        qs, cols, d = [str(x) for x in z["queries"]], [str(x) for x in z["cols"]], z["d"]
    arm = {"published_sha": sha, "published_masked_sha": shm, "rows": {}}
    for i, q in enumerate(qs):
        m = [j for j, c in enumerate(cols) if c != q]
        r = rel(d[i, m], pub.loc[q, [cols[j] for j in m]].to_numpy(float))
        arm["rows"][q] = {"n": len(m), "max_rel": float(r.max()),
                          "rank": ranking_check(q, [cols[j] for j in m], d[i, m], pubm, pop, meta,
                                                TOL["sampler"])}
    res["arms"]["sampler"] = arm
    # Yottixel
    pub, sha = published("yottixel")
    pubm, shm = published("yottixel", True)
    arm = {"published_sha": sha, "published_masked_sha": shm, "rows": {}}
    for q in sel:
        r = {}
        p = base / f"yottixel/c0_cpu_{q}.npz"
        if p.exists():
            with np.load(p) as z:
                cols, fwd = [str(x) for x in z["cols"]], z["fwd"]
            pr = pub.loc[q, cols].to_numpy(float)
            pc = pub.loc[cols, q].to_numpy(float)
            r.update(n=len(cols), cpu_vs_pub_row_mismatch=int((fwd != pr).sum()),
                     cpu_vs_pub_col_mismatch=int((fwd != pc).sum()),
                     rank=ranking_check(q, cols, fwd, pubm, pop, meta, 0.0))
            cpu = dict(zip(cols, fwd))
        else:
            r["cpu"] = "MISSING"
            cpu = None
        p = base / f"yottixel/c0_gpu_{q}.npz"
        if p.exists():
            with np.load(p) as z:
                gcols, gf, gr = [str(x) for x in z["cols"]], z["fwd"], z["rev"]
            m = [j for j, c in enumerate(gcols) if c != q]
            gc = [gcols[j] for j in m]
            if cpu is not None:
                r["gpu_fwd_vs_cpu_mismatch"] = int(sum(gf[j] != cpu[gcols[j]] for j in m))
            r["gpu_fwd_vs_pub_row_mismatch"] = int((gf[m] != pub.loc[q, gc].to_numpy(float)).sum())
            r["gpu_rev_vs_pub_col_mismatch"] = int((gr[m] != pub.loc[gc, q].to_numpy(float)).sum())
        else:
            r["gpu"] = "MISSING"
        arm["rows"][q] = r
    res["arms"]["yottixel"] = arm
    # OT
    pub, sha = published("ot")
    pubm, shm = published("ot", True)
    arm = {"published_sha": sha, "published_masked_sha": shm, "rows": {}}
    for q in sel:
        cols, d, code, secs, nref = [], [], [], [], []
        files = sorted(glob.glob(str(base / f"ot/c0_{q}_*.npz")))
        for f in files:
            with np.load(f) as z:
                cols += [str(x) for x in z["cols"]]
                d.append(z["d"])
                code.append(z["code"])
                secs.append(z["secs"])
                nref.append(z["n_ref"])
        if len(files) != 8:
            arm["rows"][q] = {"blocks": len(files), "status": "MISSING"}
            continue
        d, code, secs = np.concatenate(d), np.concatenate(code), np.concatenate(secs)
        r = rel(d, pub.loc[q, cols].to_numpy(float))
        arm["rows"][q] = {"n": len(cols), "blocks": 8, "max_rel": float(np.nanmax(r)),
                          "n_nonfinite": int((~np.isfinite(d)).sum()),
                          "codes": {str(k): int(v) for k, v in zip(*np.unique(code, return_counts=True))},
                          "pair_s_median": float(np.median(secs)), "pair_s_max": float(secs.max()),
                          "rank": ranking_check(q, cols, d, pubm, pop, meta, TOL["ot"])}
    res["arms"]["ot"] = arm
    # verdicts
    v = {}
    s = res["arms"]["sampler"]["rows"]
    v["sampler"] = all(x["max_rel"] <= TOL["sampler"] and x.get("rank", {}).get("status", "FAIL") != "FAIL"
                       for x in s.values()) and len(s) == len(sel)
    y = res["arms"]["yottixel"]["rows"]
    v["yottixel"] = all(x.get("cpu_vs_pub_row_mismatch") == 0 and x.get("gpu_fwd_vs_cpu_mismatch") == 0
                        and x.get("gpu_rev_vs_pub_col_mismatch") == 0
                        and x.get("rank", {}).get("status", "FAIL") != "FAIL" for x in y.values())
    o = res["arms"]["ot"]["rows"]
    v["ot"] = all(x.get("max_rel", 1) <= TOL["ot"] and x.get("n_nonfinite") == 0
                  and set(x.get("codes", {"x": 1})) == {"1"} and x.get("rank", {}).get("status", "FAIL") != "FAIL"
                  for x in o.values())
    res["verdict"] = v
    res["code"] = L.production_shas()
    L.dump_json(Path(out) / "c0_summary.json", res)
    print(json.dumps(v))
    lines = ["| Query | SAMPLER max rel | rank | Yottixel CPU≠pub (row/col) | GPU≠CPU | GPU rev≠pub col | rank | OT max rel | OT codes | rank |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for q in sel:
        a, b, c = s.get(q, {}), y.get(q, {}), o.get(q, {})
        lines.append("| %s | %.2e | %s | %s/%s | %s | %s | %s | %s | %s | %s |" % (
            q, a.get("max_rel", float("nan")), a.get("rank", {}).get("status"),
            b.get("cpu_vs_pub_row_mismatch"), b.get("cpu_vs_pub_col_mismatch"),
            b.get("gpu_fwd_vs_cpu_mismatch"), b.get("gpu_rev_vs_pub_col_mismatch"),
            b.get("rank", {}).get("status"),
            "%.2e" % c["max_rel"] if "max_rel" in c else c.get("status"), c.get("codes"),
            c.get("rank", {}).get("status")))
    (Path(out) / "c0_summary.md").write_text("\n".join(lines) + "\n\nVerdict: " + json.dumps(v) + "\n")
    return 0


def population(q, meta, cols):
    """Indices of the evaluation population for q: bank slides minus self and same-patient."""
    pq = meta[q][0]
    return [i for i, c in enumerate(cols) if c != q and meta[c][0] != pq]


def hit5_bounds(d, correct):
    """hit@5 with tie bounds: (lower, upper). d, correct over the population."""
    d = np.asarray(d, float)
    correct = np.asarray(correct, bool)
    d5 = np.sort(d)[4]
    S, T = d < d5, d == d5
    k = 5 - int(S.sum())
    upper = bool(correct[S | T].any())
    lower = bool(correct[S].any()) or int((~correct[T]).sum()) < k
    return lower, upper


def top5_set(cols, d):
    """Top-5 set with ties at the 5th distance kept (so it is order-independent)."""
    d = np.asarray(d, float)
    d5 = np.sort(d)[4]
    return {cols[i] for i in np.flatnonzero(d <= d5)}


def c1(out):
    meta = L.meta()
    base = L.STAGE_C / "c1"
    rec = json.loads((base / "sampler_rows.json").read_text())
    with np.load(base / "sampler_rows.npz") as z:
        qs, cols, d, ex = [str(x) for x in z["queries"]], [str(x) for x in z["cols"]], z["d"], z["excluded"]
    rows = {}
    for i, q in enumerate(qs):
        pop = population(q, meta, cols)
        exp_ex = np.ones(len(cols), bool)
        exp_ex[pop] = False
        rows[q] = {"n_population": len(pop), "n_excluded": int(ex[i].sum()),
                   "excluded_matches_rule": bool((ex[i] == exp_ex).all()),
                   "finite": bool(np.isfinite(d[i]).all()), "min_d": float(d[i, pop].min())}
    pr = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(L.STAGE_C / "prep/records/*.json")))]
    per = [x for r in pr for x in r["slides"]]
    t_s = np.array([x["t_sampler"] for x in per])
    t_l = np.array([x["t_load"] for x in per])
    res = {"n_library": len(cols), "n_queries": len(qs), "rows": rows,
           "rows_ok": all(r["excluded_matches_rule"] and r["finite"] for r in rows.values()),
           "timing": {k: rec[k] for k in ("pool_or_load_s", "rows_s", "full_pdist_s", "vmhwm_gb")},
           "host": rec["env"]["host"], "vector_len": rec["vector_len"], "vector_dtype": rec["vector_dtype"],
           "prep": {"n_slides": len(per), "sum_t_sampler_s": float(t_s.sum()), "sum_t_load_s": float(t_l.sum()),
                    "max_t_sampler_s": float(t_s.max()),
                    "max_shard_vmhwm_gb": float(max(r["vmhwm_gb"] for r in pr))}}
    res["projection"] = {
        "description": "Stage D SAMPLER: per-slide pooling (measured, summed over all 2,778 slides in the prep array) "
                       "+ one full 2,778 x 2,778 float64 cityblock pdist (measured once on 1 core)",
        "cpu_h": (res["prep"]["sum_t_sampler_s"] + res["prep"]["sum_t_load_s"] + rec["full_pdist_s"]) / 3600}
    L.dump_json(Path(out) / "c1_summary.json", res)
    print(json.dumps({k: res[k] for k in ("rows_ok", "timing", "projection")}, default=str))
    return 0


def c2(out):
    meta = L.meta()
    sel = [x["slide"] for x in L.selection()["queries"]]
    base = L.STAGE_C / "c2"
    res = {"exactness": {}, "sensitivity": {}, "timing": {}}
    rows = {}
    for v in "AB":
        for q in sel:
            with np.load(base / f"rows_{v}_{q}.npz") as z:
                rows[(v, q)] = ([str(x) for x in z["cols"]], z["fwd"].astype(float), z["rev"].astype(float))
    # exactness: GPU variant-A forward rows vs the production CPU hamming_min_median
    ok_all = True
    for q in sel:
        cols, fwd, _ = rows[("A", q)]
        g = dict(zip(cols, fwd))
        e = {}
        for kind, p in (("full", base / f"cpuref_{q}.npz"), ("cols30", base / f"ref30/cpuref_{q}.npz")):
            if not p.exists():
                continue
            with np.load(p) as z:
                cc, cf = [str(x) for x in z["cols"]], z["fwd"].astype(float)
            mm = int(sum(g[c] != x for c, x in zip(cc, cf)))
            e[kind] = {"n": len(cc), "mismatch": mm}
            ok_all &= mm == 0
        if "cols30" not in e:
            ok_all = False
            e["cols30"] = "MISSING"
        rows_missing = ("full" not in e) and q in ("7469", "19161", "5517")
        if rows_missing:
            ok_all = False
            e["full"] = "MISSING"
        res["exactness"][q] = e
    res["exact_all"] = bool(ok_all)
    # sign sensitivity: bit flips over the shared tiles (prep records, all slides)
    pr = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(L.STAGE_C / "prep/records/*.json")))]
    per = [x for r in pr for x in r["slides"]]
    n_shared = sum(x["n_shared"] for x in per)
    hist = np.zeros(max(len(x["flip_hist"]) for x in per), np.int64)
    for x in per:
        hist[:len(x["flip_hist"])] += np.asarray(x["flip_hist"], np.int64)  # [] for slides without shared tiles
    fb = sum(x["flip_bits_total"] for x in per)
    res["flips"] = {"n_slides": len(per), "n_code0": sum(x["n_code0"] for x in per), "n_shared_tiles": n_shared,
                    "flip_bits_total": int(fb), "bit_flip_rate": fb / (n_shared * L.NBITS),
                    "tiles_with_flip": int(sum(x["flip_tiles"] for x in per)),
                    "frac_tiles_with_flip": sum(x["flip_tiles"] for x in per) / n_shared,
                    "flips_per_tile_hist": hist.tolist(), "max_flips_per_tile": int(max(x["flip_max"] for x in per)),
                    "code1_equal_full_bank_all": all(x["code1_equal_full_bank"] for x in per
                                                     if x["code1_equal_full_bank"] is not None),
                    "code1_equal_full_bank_n_checked": sum(x["code1_equal_full_bank"] is not None for x in per),
                    "code1_equal_full_bank_n_na_no_full_bank": sum(x["code1_equal_full_bank"] is None for x in per)}
    # A vs B on the canary rows (population: bank minus self and same-patient)
    n_chg = {"top5_set_changed": 0, "hit5_decision_changed": 0, "hit5_indeterminate_either": 0}
    for q in sel:
        ca, da, _ = rows[("A", q)]
        cb, db, _ = rows[("B", q)]
        if ca != cb:
            raise ValueError(f"{q}: A/B column order differs")
        pop = population(q, meta, ca)
        pc = [ca[i] for i in pop]
        a, b = da[pop], db[pop]
        corr = np.array([meta[c][1] == meta[q][1] for c in pc])
        la, ua = hit5_bounds(a, corr)
        lb, ub = hit5_bounds(b, corr)
        dd = np.abs(a - b)
        s5a, s5b = top5_set(pc, a), top5_set(pc, b)
        det = la == ua and lb == ub
        r = {"n_population": len(pop), "absdiff_max": float(dd.max()), "absdiff_mean": float(dd.mean()),
             "frac_cols_changed": float((dd > 0).mean()), "top5_A": sorted(s5a, key=int), "top5_B": sorted(s5b, key=int),
             "top5_set_changed": s5a != s5b, "hit5_A": [la, ua], "hit5_B": [lb, ub],
             "hit5_decision_changed": bool(det and la != lb), "hit5_indeterminate_either": not det}
        for k in n_chg:
            n_chg[k] += int(r[k])
        res["sensitivity"][q] = r
    res["sensitivity_counts"] = n_chg
    # timing (GPU rows) + projection to the full 2,778 x 2,778 matrix
    import pandas as pd
    sizes = dict(pd.read_csv(L.STAGE_B / "bank_per_slide.csv", dtype={"slide": str})[["slide", "n_rows"]]
                 .itertuples(index=False))
    tj = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(base / "rows_[AB]_g*.json")))]
    pts = [(sizes[q], x["row_s"]) for t in tj for q, x in t["queries"].items()]
    n = np.array([p[0] for p in pts], float)
    t = np.array([p[1] for p in pts], float)
    slope = float((n * t).sum() / (n * n).sum())
    tot_tiles = float(sum(sizes[s] for s in L.bank_slides()))
    res["timing"] = {"tasks": [{k: x[k] for k in ("variant", "gpu", "load_s", "vmhwm_gb", "gpu_mem_peak_gb")}
                               | {"host": x["env"]["host"], "queue": x["env"].get("lsf_queue"),
                                  "rows_s": sum(r["row_s"] for r in x["queries"].values())} for x in tj],
                     "s_per_query_tile": slope, "library_tiles": tot_tiles,
                     "projected_full_matrix_gpu_h": slope * tot_tiles / 3600,
                     "note": "one GPU row = one query vs all 2,778 slides, both directions; full matrix = 2,778 rows"}
    for kind, p in (("full", base / "cpuref_full.json"),):
        if p.exists():
            x = json.loads(p.read_text())
            res["timing"]["cpu_full_rows"] = {k: x[k] for k in ("queries", "n_pairs", "seconds", "vmhwm_gb")} | {"host": x["env"]["host"]}
    L.dump_json(Path(out) / "c2_summary.json", res)
    print(json.dumps({"exact_all": res["exact_all"], "flips": res["flips"], "counts": n_chg,
                      "gpu_h": res["timing"]["projected_full_matrix_gpu_h"]}, default=str))
    return 0


def c3(out):
    base = L.STAGE_C / "c3"
    res = {"pairs": [], "gpucost": [], "rows": {}}
    for f in sorted(glob.glob(str(base / "pairs/pair_*.json"))):
        x = json.loads(Path(f).read_text())
        res["pairs"].append({k: x.get(k) for k in (
            "a", "b", "n_a", "n_b", "cells", "cap", "status", "result_code", "cost", "warnings",
            "t_load_norm", "t_read_M", "t_cost", "t_solve", "vmhwm_after_load_gb", "vmhwm_after_cost_gb",
            "vmhwm_after_solve_gb", "ru_maxrss_gb", "m_from", "error")} | {"host": x["env"]["host"],
                                                                            "file": Path(f).name})
    cpu = {(p["a"], p["b"]): p for p in res["pairs"] if p["cap"] == 100000000 and not p["m_from"]}
    sym = []
    for (a, b), p in cpu.items():
        if (b, a) in cpu and a < b:
            q = cpu[(b, a)]
            if p["status"] == q["status"] == "accepted":
                sym.append({"a": a, "b": b, "cells": p["cells"], "rel_diff": abs(p["cost"] - q["cost"]) / abs(q["cost"])})
    res["orientation_consistency"] = sym
    for f in sorted(glob.glob(str(base / "gpucost/gpucost_*.json"))):
        x = json.loads(Path(f).read_text())
        r = {k: x[k] for k in x if k not in ("env", "code")} | {"host": x["env"]["host"]}
        gm = [p for p in res["pairs"] if p["a"] == x["a"] and p["b"] == x["b"] and p["m_from"]]
        c = cpu.get((x["a"], x["b"]))
        if gm and c and c["status"] == "accepted" and gm[0]["status"] == "accepted":
            r["cost_rel_diff_gpuM_vs_cpuM"] = abs(gm[0]["cost"] - c["cost"]) / abs(c["cost"])
            r["within_tol_1e-6"] = r["cost_rel_diff_gpuM_vs_cpuM"] <= 1e-6
            r["t_total_cpu_path"] = c["t_cost"]
            r["t_total_gpu_path"] = x["t_h2d_feats"] + x["t_cost_gpu"] + x["t_d2h_M"] + x["t_write_M"] + gm[0]["t_read_M"]
            r["solve_s_cpuM"], r["solve_s_gpuM"] = c["t_solve"], gm[0]["t_solve"]
        res["gpucost"].append(r)
    recs = []
    for f in sorted(glob.glob(str(base / "rows/row_*.jsonl"))):
        for ln in Path(f).read_text().splitlines():
            recs.append(json.loads(ln))
    last = {}
    for r in recs:
        last[(r["query"], r["ref"])] = r
    attempts = {}
    for r in recs:
        attempts[(r["query"], r["ref"])] = attempts.get((r["query"], r["ref"]), 0) + 1
    by_q = {}
    for (q, ref), r in last.items():
        by_q.setdefault(q, []).append(r)
    for q, rs in by_q.items():
        st = {}
        for r in rs:
            st[r["status"]] = st.get(r["status"], 0) + 1
        acc = [r for r in rs if r["status"] == "accepted"]
        res["rows"][q] = {"n_refs": len(rs), "status": st,
                          "n_retried": sum(1 for r in rs if attempts[(q, r["ref"])] > 1),
                          "solve_s_sum": float(sum(r["t_solve"] for r in acc)),
                          "cost_s_sum": float(sum(r["t_cost"] for r in acc)),
                          "child_maxrss_gb_max": float(max((r["child_maxrss_gb"] for r in acc), default=0)),
                          "cells_max": int(max((r["cells"] for r in acc), default=0))}
    acc = [r for r in last.values() if r["status"] == "accepted"] + [
        p for p in cpu.values() if p["status"] == "accepted"]
    if acc:
        c = np.array([r["cells"] for r in acc], float)
        ts = np.array([r["t_solve"] + r["t_cost"] for r in acc], float)
        rss = np.array([r.get("child_maxrss_gb") or r.get("vmhwm_after_solve_gb") for r in acc], float)
        m = c >= 1e6
        p_, a_ = np.polyfit(np.log(c[m]), np.log(ts[m]), 1)
        resid = np.log(ts[m]) - (a_ + p_ * np.log(c[m]))
        k_rss = np.polyfit(c, rss, 1)
        res["fit"] = {"n": int(m.sum()), "time_s": f"exp({a_:.4f}) * cells^{p_:.4f}", "exponent": float(p_),
                      "log_intercept": float(a_), "resid_log_q05_q50_q95": np.quantile(resid, [.05, .5, .95]).tolist(),
                      "rss_gb_linear": {"per_gcell": float(k_rss[0] * 1e9), "intercept": float(k_rss[1])},
                      "rss_gb_per_cell_max": float((rss / c)[c >= 1e7].max()) if (c >= 1e7).any() else None}
        import pandas as pd
        sizes = pd.read_csv(L.STAGE_B / "bank_per_slide.csv", dtype={"slide": str}).set_index("slide")["n_rows"]
        n = sizes.loc[L.bank_slides()].to_numpy(float)
        iu = np.triu_indices(len(n), 1)
        cells = n[iu[0]] * n[iu[1]]
        tpred = np.exp(a_) * cells ** p_
        q95 = float(np.exp(np.quantile(resid, .95)))
        res["projection"] = {"n_pairs_unordered": int(len(cells)), "cells_total": float(cells.sum()),
                             "cpu_h_point": float(tpred.sum() / 3600), "cpu_h_resid_q95": float(tpred.sum() * q95 / 3600),
                             "pair_cells_quantiles": dict(zip(["p50", "p90", "p99", "max"], map(float, np.quantile(cells, [.5, .9, .99, 1])))),
                             "max_pair_pred_h": float(tpred.max() / 3600)}
    L.dump_json(Path(out) / "c3_summary.json", res)
    print(json.dumps({k: res.get(k) for k in ("fit", "projection", "orientation_consistency")}, default=str)[:3000])
    return 0


def mp(out):
    """Median-pooling arm: C0 (vs the published mosaic matrix) and C1 (100 % rows, prep cross-checks)."""
    sel = [x["slide"] for x in L.selection()["queries"]]
    meta = L.meta()
    pop = set(L.bank_slides())
    pub, sha = published("median_pooling")
    pubm, shm = published("median_pooling", True)
    base = L.STAGE_C / "c0" / "median_pooling"
    rec0 = json.loads((base / "median_pooling_c0.json").read_text())
    with np.load(base / "median_pooling_c0.npz") as z:
        qs, cols, d = [str(x) for x in z["queries"]], [str(x) for x in z["cols"]], z["d"]
    rows0 = {}
    for i, q in enumerate(qs):
        m = [j for j, c in enumerate(cols) if c != q]
        r = rel(d[i, m], pub.loc[q, [cols[j] for j in m]].to_numpy(float))
        rows0[q] = {"n": len(m), "max_rel": float(r.max()),
                    "rank": ranking_check(q, [cols[j] for j in m], d[i, m], pubm, pop, meta, TOL["median_pooling"])}
    v0 = (len(rows0) == len(sel) and set(qs) == set(sel)
          and all(x["max_rel"] <= TOL["median_pooling"] and x["rank"]["status"] != "FAIL" for x in rows0.values()))
    res0 = {"queries": sel, "tolerance_max_rel": TOL["median_pooling"],
            "published": str(L.PUBLISHED / "combined_similarity_matrix_uni2_median_pooling.pkl"),
            "published_sha": sha, "published_masked_sha": shm, "rows": rows0, "pass": bool(v0),
            "c0_record": {k: rec0[k] for k in ("n_library", "vector_dtype", "vector_len", "pool_or_load_s", "rows_s")},
            "code": L.production_shas()}
    c0dir = L.STAGE_C / "c0"
    L.dump_json(c0dir / "c0_median_pooling_summary.json", res0)
    lines = ["| Query | median-pooling max rel | rank |", "|---|---|---|"]
    lines += ["| %s | %.2e | %s |" % (q, rows0[q]["max_rel"], rows0[q]["rank"]["status"]) for q in sel if q in rows0]
    (c0dir / "c0_median_pooling_summary.md").write_text("\n".join(lines) + f"\n\nPASS: {bool(v0)}\n")
    # C1 (only once its rows exist; C0 is gated first)
    c1dir = L.STAGE_C / "c1"
    if not (c1dir / "median_pooling_rows.json").exists():
        print(json.dumps({"c0_pass": bool(v0), "c0_max_rel": max(x["max_rel"] for x in rows0.values()),
                          "c1": "not yet run"}))
        return 0 if v0 else 2
    rec = json.loads((c1dir / "median_pooling_rows.json").read_text())
    with np.load(c1dir / "median_pooling_rows.npz") as z:
        qs, cols, d, ex = [str(x) for x in z["queries"]], [str(x) for x in z["cols"]], z["d"], z["excluded"]
    rows = {}
    for i, q in enumerate(qs):
        popi = population(q, meta, cols)
        exp_ex = np.ones(len(cols), bool)
        exp_ex[popi] = False
        rows[q] = {"n_population": len(popi), "n_excluded": int(ex[i].sum()),
                   "excluded_matches_rule": bool((ex[i] == exp_ex).all()),
                   "finite": bool(np.isfinite(d[i]).all()), "min_d": float(d[i, popi].min())}
    pr = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(L.STAGE_C / "prep/records_median/*.json")))]
    per = [x for r in pr for x in r["slides"]]
    nshard = len(glob.glob(str(L.STAGE_C / "prep/median_pooling/*.npz")))
    t_m = np.array([x["t_median"] for x in per])
    t_l = np.array([x["t_load"] for x in per])
    res = {"n_library": len(cols), "n_queries": len(qs), "rows": rows,
           "rows_ok": bool(cols == L.bank_slides() and all(r["excluded_matches_rule"] and r["finite"] for r in rows.values())),
           "timing": {k: rec[k] for k in ("pool_or_load_s", "rows_s", "full_pdist_s", "vmhwm_gb")},
           "host": rec["env"]["host"], "vector_len": rec["vector_len"], "vector_dtype": rec["vector_dtype"],
           "prep": {"n_slides": len(per), "n_slides_unique": len({x["slide"] for x in per}),
                    "n_shards_records": len(pr), "n_shards_vectors": nshard,
                    "sampler_equal_prep_all": all(x["sampler_equal_prep"] for x in per),
                    "rows_total": int(sum(x["n_rows"] for x in per)),
                    "sum_t_median_s": float(t_m.sum()), "sum_t_load_s": float(t_l.sum()),
                    "max_shard_vmhwm_gb": float(max(r["vmhwm_gb"] for r in pr))}}
    res["prep_ok"] = bool(res["prep"]["n_slides_unique"] == len(L.bank_slides()) == res["prep"]["n_slides"]
                          and res["prep"]["sampler_equal_prep_all"])
    res["projection"] = {"description": "Stage D median pooling: per-slide pooling (measured, prep array) + one full "
                                        "2,778 x 2,778 float64 cityblock pdist (measured once)",
                         "cpu_h": (res["prep"]["sum_t_median_s"] + res["prep"]["sum_t_load_s"] + rec["full_pdist_s"]) / 3600}
    L.dump_json(c1dir / "c1_median_pooling_summary.json", res)
    print(json.dumps({"c0_pass": bool(v0), "c0_max_rel": max(x["max_rel"] for x in rows0.values()),
                      "rows_ok": res["rows_ok"], "prep_ok": res["prep_ok"], "prep": res["prep"],
                      "projection": res["projection"]}, default=str))
    return 0 if (v0 and res["rows_ok"] and res["prep_ok"]) else 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["c0", "c1", "c2", "c3", "mp"])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = a.out or str(L.STAGE_C / a.mode)
    Path(out).mkdir(parents=True, exist_ok=True)
    return {"c0": c0, "c1": c1, "c2": c2, "c3": c3, "mp": mp}[a.mode](out)


if __name__ == "__main__":
    sys.exit(main())
