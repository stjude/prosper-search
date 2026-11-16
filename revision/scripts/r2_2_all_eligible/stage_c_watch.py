#!/usr/bin/env python3
"""R2.2-ALL Stage C persistent completion-only watcher (generalised from stage_b_b3_watch.py).

Reads the pool list from --pools-file every cycle ("<pool> cpu|gpu" per line), so pools created
later in the Stage C chain are adopted without a restart. For every listed, unfinished pool it
applies the B3 reconciliation between lsfq cycles (no runner holds the pool meanwhile), then runs
``lsfq pool run <pool> --once``:

1. GPU pools: rhel88_gpu blacklisted; nodegpu247/248 and any host whose task log reports the
   wrapper's FOREIGN_GPU_OCCUPANCY guard (exit 75) are excluded in ``select[]``.
2. One LSF job per task (lsfq's stale-marker orphans are killed/adopted; stale markers removed).
3. EXITs that were not the task's fault are forgiven (attempt restored, task back to backlog):
   job never ran, CUDA OOM, guard exit 75, wrong GPU/OS refusal (exit 3), false EXITs whose job
   later completed rc=0. A genuine failure keeps lsfq's attempt accounting.

Output: one line per pool when it completes (completion-only); details go to --log.
Exits when every listed pool is complete and --final-flag exists.
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
GPU_SELECT = "a100_80g && rhel8"
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
        if len(p) == 4 and p[0].isdigit() and p[3].startswith(f"lsfq_{pool}_"):
            jobs.append({"job": p[0], "stat": p[1], "queue": p[2], "tid": p[3].rsplit("_", 1)[-1]})
    return jobs


def task_log(joblogs, pool, tid, job):
    p = os.path.join(joblogs, f"lsfq_{pool}_{tid}.{job}.log")
    return open(p, errors="replace").read() if os.path.exists(p) else None


def classify_exit(text):
    if text is None:
        return "never_ran"
    low = text.lower()
    if GUARD in text:
        return "guard_foreign_occupancy"
    if "cuda out of memory" in low or "outofmemoryerror" in low:
        return "cuda_oom"
    if "is not admitted" in text:
        return "wrong_gpu_or_os"
    m = re.findall(r"\[task\] rc=(\d+)", text)
    if m and m[-1] == "0":
        return "false_exit_completed_ok"
    if not m:
        return "killed_or_limit"  # TERM_* (memlimit/runlimit) or killed: lsfq bumps and retries
    return "genuine"


def bad_hosts(joblogs, pool):
    hosts = set(STATIC_BAD)
    for p in glob.glob(os.path.join(joblogs, f"lsfq_{pool}_*.log")):
        try:
            txt = open(p, errors="replace").read()
        except OSError:
            continue
        m = re.search(r"FOREIGN_GPU_OCCUPANCY \S+ on (\S+) gpu", txt)
        if m:
            hosts.add(m.group(1))
    return sorted(hosts)


def correct(st, pool_dir, joblogs, kind, fh):
    pol, tasks, name = st["policy"], st["tasks"], st["name"]
    if kind == "gpu":
        qs = [q for q in st["filters"]["queues"].split(",") if q and q != "rhel88_gpu"]
        st["filters"]["queues"] = ",".join(qs)
        st["blacklist"]["rhel88_gpu"] = BLACKLIST_REASON
        st["trusted"] = [q for q in st["trusted"] if q != "rhel88_gpu"]
        extra = "-R 'select[%s && %s]'" % (GPU_SELECT, " && ".join(
            f"hname!={h}" for h in bad_hosts(joblogs, name)))
        if st["shape"].get("extra") != extra:
            log(fh, f"{name}: select updated: {extra}")
            st["shape"]["extra"] = extra
    by_tid = {}
    for j in active_jobs(name):
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
            keep = tracked[0] if tracked and (tracked[0]["stat"] == "RUN" or not run) else (run or js)[0]
        for j in js:
            if keep is None or j["job"] != keep["job"]:
                log(fh, f"{name}: kill duplicate job {j['job']} ({j['stat']} {j['queue']}) task {tid}")
                subprocess.run(["bkill", j["job"]], capture_output=True, timeout=60)
        if keep is not None:
            if t.get("job") != keep["job"] or t["state"] != "active":
                log(fh, f"{name}: adopt job {keep['job']} ({keep['stat']}) for task {tid} "
                        f"[was {t['state']} job {t.get('job')}]")
                t.update(state="active", job=keep["job"], queue=keep["queue"],
                         started_seen=keep["stat"] == "RUN")
                t.setdefault("history", []).append(
                    {"job": keep["job"], "queue": keep["queue"], "end": now_s(),
                     "note": "adopted by stage_c_watch (lsfq stale-marker orphan)"})
            mk = os.path.join(pool_dir, "markers", tid + ".rc")
            if os.path.exists(mk):
                os.remove(mk)
                log(fh, f"{name}: removed stale marker of task {tid}")
    for tid, t in sorted(tasks.items()):
        for h in t.get("history", []):
            if not str(h.get("note", "")).startswith("EXIT") or "exit_class" in h:
                continue
            cls = classify_exit(task_log(joblogs, name, tid, h["job"]))
            h["exit_class"] = cls
            if cls not in ("genuine", "killed_or_limit"):
                t["attempts"] = max(0, t["attempts"] - 1)
            log(fh, f"{name}: task {tid} job {h['job']} EXIT classified {cls}")
        ok = [h["job"] for h in t.get("history", []) if h.get("exit_class") == "false_exit_completed_ok"]
        if t["state"] in ("failed", "backlog") and ok:
            log(fh, f"{name}: task {tid} -> done (job {ok[-1]} completed rc=0)")
            t["state"], t["job"] = "done", None
            continue
        if t["state"] == "failed" and t["attempts"] < pol["max_attempts"]:
            log(fh, f"{name}: task {tid} failed -> backlog (attempts {t['attempts']})")
            t["state"], t["job"] = "backlog", None


def counts_of(st):
    c = {}
    for t in st["tasks"].values():
        c[t["state"]] = c.get(t["state"], 0) + 1
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools-file", required=True)
    ap.add_argument("--joblogs", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--runner-log", required=True)
    ap.add_argument("--final-flag", required=True)
    ap.add_argument("--interval", type=int, default=90)
    ap.add_argument("--lsfq", default=os.path.expanduser(
        "~/.claude/skills/lsf-max-throughput/scripts/lsfq.py"))
    a = ap.parse_args(argv)
    done = set()
    with open(a.log, "a") as fh:
        log(fh, f"watcher start pid {os.getpid()}")
        while True:
            pools = []
            for ln in open(a.pools_file):
                p = ln.split()
                if len(p) == 2 and not ln.startswith("#"):
                    pools.append((p[0], p[1]))
            pending = 0
            for name, kind in pools:
                pool_dir = os.path.expanduser(f"~/.lsfq/pools/{name}")
                state = os.path.join(pool_dir, "state.json")
                if not os.path.exists(state):
                    pending += 1
                    continue
                st = json.load(open(state))
                correct(st, pool_dir, a.joblogs, kind, fh)
                tmp = state + f".tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump(st, f, indent=1)
                os.replace(tmp, state)
                c = counts_of(st)
                if c.get("backlog", 0) == 0 and c.get("active", 0) == 0:
                    if name not in done:
                        done.add(name)
                        msg = f"POOL_COMPLETE {name} {json.dumps(c, sort_keys=True)}"
                        log(fh, msg)
                        print(f"{now_s()} {msg}", flush=True)
                    continue
                done.discard(name)
                pending += 1
                with open(a.runner_log, "a") as rl:
                    rl.write(f"--- {now_s()} {name}\n")
                    rl.flush()
                    subprocess.run([sys.executable, a.lsfq, "pool", "run", name, "--once"],
                                   stdout=rl, stderr=subprocess.STDOUT)
            if pending == 0 and os.path.exists(a.final_flag):
                log(fh, "all pools complete and final flag present; exit")
                print(f"{now_s()} WATCHER_EXIT", flush=True)
                return 0
            time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
