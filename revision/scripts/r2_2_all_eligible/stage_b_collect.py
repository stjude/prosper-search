#!/usr/bin/env python3
"""R2.2-ALL Stage B accounting/collection (B5): LSF accounting, per-queue counts, retries, tiles/s.

Reads lsfq pool states (job ids), one ``bacct -l`` call over all of them, B3 per-slide json
timing records and the B3 task logs; writes ``<out>/accounting.json`` and per-job/per-slide CSVs.
Read-only on everything except ``--out``.
"""
import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


def pool_jobs(name):
    st = json.load(open(os.path.expanduser(f"~/.lsfq/pools/{name}/state.json")))
    rows = []
    for tid, t in st["tasks"].items():
        for h in t.get("history", []):
            if h.get("job"):
                rows.append({"pool": name, "tid": tid, "job": str(h["job"]), "note": h.get("note"),
                             "exit_class": h.get("exit_class")})
        if t.get("job"):
            rows.append({"pool": name, "tid": tid, "job": str(t["job"]), "note": "current",
                         "exit_class": None})
    return st, rows


def bacct(jobs):
    out = subprocess.run(["bacct", "-l", *jobs], capture_output=True, text=True).stdout
    recs = {}
    for blk in out.split("-" * 78):
        m = re.search(r"Job <(\d+)>", blk)
        if not m:
            continue
        flat = re.sub(r"\n\s{21}", "", blk)
        r = {"job": m.group(1)}
        q = re.search(r"Queue <([^>]+)>", flat)
        r["queue"] = q.group(1) if q else None
        h = re.search(r"Dispatched \d+ Task\(s\) on Host\(s\) <(?:\d+\*)?([^>]+)>", flat)
        r["host"] = h.group(1) if h else None
        s = re.search(r"CPU_T\s+WAIT\s+TURNAROUND\s+STATUS.*\n\s*([\d.]+)\s+(\d+)\s+(\d+)\s+(\w+)", flat)
        if s:
            r.update(cpu_s=float(s.group(1)), wait_s=int(s.group(2)), turnaround_s=int(s.group(3)),
                     status=s.group(4))
            r["run_s"] = r["turnaround_s"] - r["wait_s"] if r["host"] else 0
        recs[r["job"]] = r
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b", required=True)
    ap.add_argument("--pools", default="r22b_b3,r22b_b4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    B, out = Path(a.stage_b), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in a.pools.split(","):
        rows += pool_jobs(p)[1]
    jobs = pd.DataFrame(rows).drop_duplicates(["pool", "job"], keep="last")
    acc = bacct(sorted(jobs.job.unique()))
    jobs = jobs.merge(pd.DataFrame(acc.values()), on="job", how="left")
    jobs.to_csv(out / "lsf_jobs.csv", index=False)

    slides = []
    for p in sorted((B / "b3/out").glob("*.json")):
        j = json.loads(p.read_text())
        env = j.get("env") or {}
        slides.append({"slide": j["slide"], "n_tiles": j["n_tiles"], "gpu_model": j["gpu_model"],
                       "host": env.get("host"), "queue": env.get("lsf_queue"),
                       "job": env.get("lsf_job"), **{k: v for k, v in j["timing"].items()}})
    sl = pd.DataFrame(slides)
    sl.to_csv(out / "b3_slide_timing.csv", index=False)

    summ = {}
    for p, g in jobs.groupby("pool"):
        ran = g[g.run_s.fillna(0) > 0]
        summ[p] = {
            "jobs_total": int(len(g)), "jobs_ran": int(len(ran)),
            "run_h": round(ran.run_s.sum() / 3600, 3), "cpu_h": round(g.cpu_s.fillna(0).sum() / 3600, 3),
            "status": g.status.fillna("unknown").value_counts().to_dict(),
            "exit_class": g.exit_class.dropna().value_counts().to_dict(),
            "done_jobs_by_queue": g[g.status == "done"].queue.value_counts().to_dict(),
            "run_h_by_queue": (ran.groupby("queue").run_s.sum() / 3600).round(3).to_dict(),
        }
    summ["r22b_b3"]["gpu_h"] = summ["r22b_b3"]["run_h"]  # 1 GPU per job (num=1/host)
    summ["b3_slides"] = {
        "n": int(len(sl)), "tiles": int(sl.n_tiles.sum()),
        "gpu_models": sl.gpu_model.value_counts().to_dict(),
        "slides_by_queue": sl.queue.value_counts().to_dict(),
        "tiles_by_queue": sl.groupby("queue").n_tiles.sum().astype(int).to_dict(),
        "tiles_per_s_quantiles_slides_ge_2000_tiles":
            sl[sl.n_tiles >= 2000].tiles_per_s.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).to_dict(),
        "tiles_per_s_median_by_host": sl.groupby("host").tiles_per_s.median().round(1).to_dict(),
        "first_finished": None, "last_finished": None,
    }
    fin = sorted(json.loads(p.read_text()).get("finished", "") for p in (B / "b3/out").glob("*.json"))
    summ["b3_slides"]["first_finished"], summ["b3_slides"]["last_finished"] = fin[0], fin[-1]
    (out / "accounting.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
