#!/usr/bin/env python3
"""R2.2-ALL Stage D close-out: measured cost record (``costs.json``) and ``SHA256SUMS.txt``.

Reads only Stage D outputs and logs:
- LSF job logs (``stage_d/joblogs/lsfq_<pool>_<task>.<job>.log``): host, queue, start, end, rc of every
  attempt, including failed or killed attempts, which keep their logs;
- Yottixel row files (``stage_d/yottixel/rows/<q>.npz``): GPU model, host, LSF job, per-reference seconds,
  query tile count;
- Yottixel task records and the matrix records of the three arms;
- the Stage C prep records, for per-slide pooling time.

Writes ``stage_d/costs.json`` and, with ``--sha``, ``stage_d/SHA256SUMS.txt`` over every file under
``stage_d/`` except the checksum file itself.
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RUN = Path(os.environ.get("PROSPER_OUT", "prosper_revision_output")) / "r2_2_uni2_all_eligible_20261113/run_001"
LOGRE = re.compile(r"lsfq_(?P<pool>r22d_[a-z_]+?)_(?P<task>\d{5})\.(?P<job>\d+)\.log$")
TASKRE = re.compile(r"\[task\] host=(\S+) job=(\d+) queue=(\S+) .*start=(\S+)")
ENDRE = re.compile(r"\[task\] rc=(-?\d+) end=(\S+)")


def ts(s):
    return dt.datetime.fromisoformat(s)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def attempts(sd):
    out = []
    for p in sorted(glob.glob(str(sd / "joblogs" / "lsfq_r22d_*.log"))):
        m = LOGRE.search(os.path.basename(p))
        if not m:
            continue
        txt = Path(p).read_text(errors="replace")
        a = {"pool": m["pool"], "task": m["task"], "job": m["job"], "log": os.path.relpath(p, sd)}
        t = TASKRE.search(txt)
        if t:
            a.update(host=t[1], queue=t[3], start=t[4])
        e = ENDRE.search(txt)
        if e:
            a.update(rc=int(e[1]), end=e[2])
            if t:
                a["wall_s"] = (ts(e[2]) - ts(t[4])).total_seconds()
        else:
            a["rc"] = None  # killed/limit (no end line), or never started
        g = re.search(r"NVIDIA [^,]+", txt)
        a["gpu"] = g[0] if g and "gpu_task=1" in txt else None
        out.append(a)
    return out


def summarize_pool(at):
    ok = [a for a in at if a.get("rc") == 0]
    walls = [a["wall_s"] for a in at if "wall_s" in a]
    starts = [ts(a["start"]) for a in at if "start" in a]
    ends = [ts(a["end"]) for a in at if "end" in a]
    return {"attempts": len(at), "attempts_rc0": len(ok), "attempts_nonzero_or_killed": len(at) - len(ok),
            "tasks_rc0": len({a["task"] for a in ok}),
            "sum_attempt_wall_s": float(sum(walls)), "first_start": min(starts).isoformat() if starts else None,
            "last_end": max(ends).isoformat() if ends else None,
            "elapsed_wall_s": (max(ends) - min(starts)).total_seconds() if starts and ends else None,
            "rc0_by_queue": dict(Counter(a["queue"] for a in ok)),
            "attempts_by_queue": dict(Counter(a.get("queue") for a in at)),
            "rc0_by_host": dict(Counter(a["host"] for a in ok))}


def yottixel_rows(sd):
    per = []
    for p in sorted(glob.glob(str(sd / "yottixel/rows/*.npz"))):
        with np.load(p) as z:
            per.append({"q": Path(p).stem, "gpu": str(z["gpu"]), "host": str(z["host"]), "job": str(z["lsf_job"]),
                        "n_tiles": int(z["n_tiles"]), "row_s": float(np.sum(z["secs"]))})
    return per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage-d", default=str(RUN / "stage_d"))
    ap.add_argument("--stage-c", default=str(RUN / "stage_c"))
    ap.add_argument("--sha", action="store_true", help="also write SHA256SUMS.txt (run last)")
    a = ap.parse_args(argv)
    sd, sc = Path(a.stage_d), Path(a.stage_c)
    at = attempts(sd)
    by_pool = defaultdict(list)
    for x in at:
        by_pool[x["pool"]].append(x)
    res = {"generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": str(Path(__file__).resolve()), "script_sha256": sha_file(__file__),
           "pools": {k: summarize_pool(v) for k, v in sorted(by_pool.items())}, "attempts": at}

    def prep_sum(pattern, key):
        recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(sc / pattern)))]
        sl = [x for r in recs for x in r["slides"]]
        return {"n_slides": len(sl), f"sum_{key}": float(sum(x[key] for x in sl)),
                "sum_t_load_s": float(sum(x["t_load"] for x in sl))}

    for arm, mfile, pat, key in (("sampler", "sampler/sampler_matrix.json", "prep/records/*.json", "t_sampler"),
                                 ("median_pooling", "median_pooling/median_pooling_matrix.json",
                                  "prep/records_median/*.json", "t_median")):
        m = json.loads((sd / mfile).read_text())
        try:
            pp = prep_sum(pat, key)
        except KeyError as exc:
            pp = {"error": f"missing key {exc}"}
        res[arm] = {"pass": m.get("pass"), "matrix_s": m.get("matrix_s"), "vmhwm_gb": m.get("vmhwm_gb"),
                    "host": (m.get("env") or {}).get("host"), "queue": (m.get("env") or {}).get("lsf_queue"),
                    "prep_pooling": pp,
                    "cpu_h_total": ((pp.get(f"sum_{key}", 0) + pp.get("sum_t_load_s", 0)) +
                                    (m.get("matrix_s") or 0)) / 3600,
                    "lsf": res["pools"].get(f"r22d_{'median' if arm == 'median_pooling' else arm}")}
    rows = yottixel_rows(sd)
    y = {"n_rows": len(rows), "gpu_models": dict(Counter(r["gpu"] for r in rows)),
         "rows_by_host": dict(Counter(r["host"] for r in rows))}
    if rows:
        tps = np.array([r["n_tiles"] / r["row_s"] for r in rows if r["row_s"] > 0])
        y.update(sum_row_compute_s=float(sum(r["row_s"] for r in rows)),
                 gpu_h_row_compute=float(sum(r["row_s"] for r in rows)) / 3600,
                 query_tiles=int(sum(r["n_tiles"] for r in rows)),
                 tiles_per_s={"median": float(np.median(tps)), "p10": float(np.percentile(tps, 10)),
                              "p90": float(np.percentile(tps, 90)), "min": float(tps.min()), "max": float(tps.max())},
                 note="tiles/s = query tiles / row seconds; one row = one query vs all 2,778 references, both directions")
    trs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(sd / "yottixel/task_records/*.json")))]
    y["task_records"] = len(trs)
    if trs:
        y["gpu_mem_peak_gb_max"] = max(t.get("gpu_mem_peak_gb", 0) for t in trs)
        y["vmhwm_gb_max"] = max(t.get("vmhwm_gb", 0) for t in trs)
        y["load_s_median"] = float(np.median([t["load_s"] for t in trs if "load_s" in t] or [np.nan]))
    # row tasks ran in two pools over the same task list: r22d_yottixel (forward order) and r22d_ysh
    # (reverse order, gpu_short); a resumed task skips rows that are already valid
    row_pools = [p for p in ("r22d_yottixel", "r22d_ysh") if p in by_pool]
    yp = summarize_pool([x for p in row_pools for x in by_pool[p]]) if row_pools else {}
    y["lsf_row_pools"] = row_pools
    y["lsf_by_pool"] = {p: res["pools"][p] for p in row_pools}
    y["lsf"] = yp
    y["gpu_h_reserved"] = yp.get("sum_attempt_wall_s", 0) / 3600  # 1 GPU per attempt
    y["rows_by_lsf_pool"] = dict(Counter(
        next((x["pool"] for x in at if x["job"] == r["job"]), "unknown") for r in rows))
    mj = sd / "yottixel/matrix/yottixel_matrix.json"
    if mj.exists():
        m = json.loads(mj.read_text())
        y["assemble"] = {"pass": m.get("pass"), "vmhwm_gb": m.get("vmhwm_gb"), "host": (m.get("env") or {}).get("host"),
                         "lsf": res["pools"].get("r22d_yasm")}
    res["yottixel"] = y
    (sd / "costs.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("attempts",)}, indent=1, default=str)[:6000])
    if a.sha:
        files = sorted(p for p in sd.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"
                       and "__pycache__" not in p.parts and not p.name.startswith("."))
        with open(sd / "SHA256SUMS.txt", "w") as f:
            for p in files:
                f.write(f"{sha_file(p)}  {p.relative_to(sd)}\n")
        print(f"SHA256SUMS.txt: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
