#!/usr/bin/env python3
"""R2.2-ALL B3 single completion watcher (analyst correction 2026-11-13 16:55 CST).

Drives the lsfq pool one cycle at a time (``lsfq pool run NAME --once``) and, between
cycles, while no runner holds the pool, applies four corrections to the pool state:

1. rhel88_gpu is blacklisted (foreign non-LSF GPU occupancy; LSF cannot see it).
2. Hosts are excluded from the pool's ``select[]`` when any task log reports
   FOREIGN_GPU_OCCUPANCY (the wrapper's free-memory guard, exit 75), plus nodegpu247/248.
3. lsfq leaves a task's old rc marker in place until the resubmitted job *starts*; a
   cycle in between retires the still-pending job as "EXIT" without killing it, leaving
   orphans that run the task twice. Every LSF job of the pool is reconciled with the task
   it belongs to: one job is kept (a running one first), the others are killed, and
   markers of tasks whose kept job is still PEND/RUN are removed as stale.
4. EXITs that were not the task's fault are forgiven (attempt count restored, failed
   task returned to the backlog): CUDA OOM, the guard's exit 75, the wrong-GPU/OS refusal
   (exit 3) and false EXITs whose job never ran or later completed rc=0.

Stops when the pool has no backlog and no active task. Writes only the lsfq pool state,
its own log and kills/adopts jobs of this pool.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

GUARD = "FOREIGN_GPU_OCCUPANCY"
STATIC_BAD = ("nodegpu247", "nodegpu248")
BLACKLIST_REASON = ("foreign non-LSF GPU occupancy (Ray workers ~80 GB per A100), invisible "
                    "to LSF -> CUDA OOM loop; lead 2026-11-13 16:55 CST")


def now_s():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(fh, msg):
    fh.write(f"{now_s()} {msg}\n")
    fh.flush()


def active_jobs(pool):
    out = subprocess.run(["bjobs", "-noheader", "-J", f"lsfq_{pool}_*", "-o",
                          "jobid stat queue job_name delimiter='|'"],
                         capture_output=True, text=True, timeout=120).stdout
    jobs = []
    for ln in out.splitlines():
        p = ln.strip().split("|")
        if len(p) == 4 and p[0].isdigit():
            jobs.append({"job": p[0], "stat": p[1], "queue": p[2],
                         "tid": p[3].rsplit("_", 1)[-1]})
    return jobs


def task_log(logs_dir, tid, job):
    hits = glob.glob(os.path.join(logs_dir, f"task_{int(tid) - 1:04d}.{job}.log"))
    return open(hits[0], errors="replace").read() if hits else None


def classify_exit(text):
    """'genuine' only for a real task failure; anything else is forgiven."""
    if text is None:
        return "never_ran"
    low = text.lower()
    if GUARD in text:
        return "guard_foreign_occupancy"
    if "out of memory" in low or "outofmemoryerror" in low:
        return "cuda_oom"
    if "is not admitted" in text or "not a RHEL8 image" in text:
        return "wrong_gpu_or_os"
    m = re.findall(r"\[task\] rc=(\d+)", text)
    if m and m[-1] == "0":
        return "false_exit_completed_ok"
    if not m:
        return "false_exit_running_or_killed"
    return "genuine"


def bad_hosts(logs_dir):
    hosts = set(STATIC_BAD)
    for p in glob.glob(os.path.join(logs_dir, "task_*.log")):
        try:
            txt = open(p, errors="replace").read()
        except OSError:
            continue
        if GUARD in txt:
            m = re.search(r"\[b3-guard\] FOREIGN_GPU_OCCUPANCY .* on (\S+) gpu", txt)
            if m:
                hosts.add(m.group(1))
    return sorted(hosts)


def correct(st, pool_dir, logs_dir, fh, kill=True):
    pol, tasks = st["policy"], st["tasks"]
    # 1. rhel88_gpu out of the pool
    qs = [q for q in st["filters"]["queues"].split(",") if q and q != "rhel88_gpu"]
    st["filters"]["queues"] = ",".join(qs)
    st["blacklist"]["rhel88_gpu"] = BLACKLIST_REASON
    st["trusted"] = [q for q in st["trusted"] if q != "rhel88_gpu"]
    # 2. host exclusions in select[]
    hosts = bad_hosts(logs_dir)
    extra = "-R 'select[a100_80g && rhel8 && %s]'" % " && ".join(f"hname!={h}" for h in hosts)
    if st["shape"].get("extra") != extra:
        log(fh, f"select updated: {extra}")
        st["shape"]["extra"] = extra
    # 3. one LSF job per task; stale markers
    by_tid = {}
    for j in active_jobs(st["name"]):
        by_tid.setdefault(j["tid"], []).append(j)
    for tid, js in sorted(by_tid.items()):
        t = tasks.get(tid)
        if t is None:
            continue
        if t["state"] == "done":
            keep = None
        else:
            run = [j for j in js if j["stat"] == "RUN"]
            tracked = [j for j in js if j["job"] == t.get("job")]
            if tracked and (tracked[0]["stat"] == "RUN" or not run):
                keep = tracked[0]
            else:
                keep = (run or js)[0]
        for j in js:
            if keep is None or j["job"] != keep["job"]:
                log(fh, f"kill duplicate job {j['job']} ({j['stat']} {j['queue']}) of task {tid} "
                        f"[{t['state']}]")
                if kill:
                    subprocess.run(["bkill", j["job"]], capture_output=True, timeout=60)
        if keep is not None:
            if t.get("job") != keep["job"] or t["state"] != "active":
                log(fh, f"adopt job {keep['job']} ({keep['stat']} {keep['queue']}) for task {tid} "
                        f"[was {t['state']} job {t.get('job')}]")
                t.update(state="active", job=keep["job"], queue=keep["queue"],
                         started_seen=keep["stat"] == "RUN")
                t.setdefault("history", []).append(
                    {"job": keep["job"], "queue": keep["queue"], "end": now_s(),
                     "note": "adopted by stage_b_b3_watch (orphan of lsfq stale-marker retire)"})
            mk = os.path.join(pool_dir, "markers", tid + ".rc")
            if os.path.exists(mk):
                os.remove(mk)
                log(fh, f"removed stale marker of task {tid} (job {keep['job']} {keep['stat']})")
    # 4. forgive EXITs that were not the task's fault
    for tid, t in sorted(tasks.items()):
        for h in t.get("history", []):
            if not str(h.get("note", "")).startswith("EXIT") or "exit_class" in h:
                continue
            cls = classify_exit(task_log(logs_dir, tid, h["job"]))
            h["exit_class"] = cls
            if cls != "genuine":
                t["attempts"] = max(0, t["attempts"] - 1)
        ok_jobs = [h["job"] for h in t.get("history", [])
                   if h.get("exit_class") == "false_exit_completed_ok"]
        if t["state"] in ("failed", "backlog") and ok_jobs:
            log(fh, f"task {tid} {t['state']} -> done (job {ok_jobs[-1]} completed rc=0; "
                    f"lsfq had retired it on a stale marker)")
            t["state"] = "done"
            t["job"] = None
            continue
        if t["state"] == "failed" and t["attempts"] < pol["max_attempts"]:
            log(fh, f"task {tid} failed -> backlog (attempts {t['attempts']})")
            t["state"] = "backlog"
            t["job"] = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", default="r22b_b3")
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--lsfq", default=os.path.expanduser(
        "~/.claude/skills/lsf-max-throughput/scripts/lsfq.py"))
    ap.add_argument("--runner-log", required=True)
    ap.add_argument("--dry-run", action="store_true", help="apply corrections in memory only")
    a = ap.parse_args(argv)
    pool_dir = os.path.expanduser(f"~/.lsfq/pools/{a.pool}")
    state = os.path.join(pool_dir, "state.json")
    with open(a.log, "a") as fh:
        log(fh, f"watcher start pid {os.getpid()} pool {a.pool} dry_run={a.dry_run}")
        while True:
            st = json.load(open(state))
            correct(st, pool_dir, a.logs_dir, fh, kill=not a.dry_run)
            counts = {}
            for t in st["tasks"].values():
                counts[t["state"]] = counts.get(t["state"], 0) + 1
            if a.dry_run:
                log(fh, f"dry-run counts after corrections {json.dumps(counts, sort_keys=True)}")
                return 0
            tmp = state + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(st, f, indent=1)
            os.replace(tmp, state)
            if counts.get("backlog", 0) == 0 and counts.get("active", 0) == 0:
                log(fh, f"pool complete {json.dumps(counts, sort_keys=True)}")
                return 0 if counts.get("failed", 0) == 0 else 1
            with open(a.runner_log, "a") as rl:
                subprocess.run([sys.executable, a.lsfq, "pool", "run", a.pool, "--once"],
                               stdout=rl, stderr=subprocess.STDOUT)
            time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
