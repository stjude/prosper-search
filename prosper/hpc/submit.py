"""Queue-aware LSF submission: canary first, measure, then fan out across queues.

The discipline this enforces
----------------------------
1. **Canary before fan-out.** Submit exactly one element, wait for it, read its
   resource epilogue. Only then size and submit the rest. A 2,921-element array
   whose every element fails on a missing input, or reserves 8x the memory it
   needs, costs a day to discover otherwise -- and that has already happened in
   this project.
2. **Right-size from measurement, never from the old scripts.** ``-M`` and
   ``rusage[mem=]`` are **per-slot** on this cluster. Proof from a real epilogue:
   a ``-n 64 -M 8192`` job reported ``Total Requested Memory 524288 MB`` against
   ``Max Memory 3827 MB``. Every element of that array asked the scheduler for
   half a terabyte. On a queue with tens of thousands of pending jobs, memory
   over-reservation is a bigger driver of time-to-completion than runtime.
3. **Spread across queues.** Never let one queue hold a whole sweep. ``standard``
   and ``priority`` share the ``rhel8_cpu`` host group, so splitting between them
   costs nothing and removes a single point of stall.
4. **Never redirect inside the job.** The existing wrappers run
   ``exec > "$STDOUT_LOG" 2> "$STDERR_LOG"`` before the real work, which swallows
   errors *and* discards LSF's resource-usage epilogue. That is why 2,921
   "can't open file" errors went unnoticed and why no timing data exists anywhere
   in the logs. Use ``-oo``/``-eo`` so LSF writes the epilogue itself.

Nothing here submits implicitly: :func:`canary` and :func:`fan_out` build the
command and only run it when ``dry_run=False``.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .queues import MAX_JOB_ARRAY_SIZE, Queue, chunk_array, rank


@dataclass
class Resources:
    slots: int = 1
    mem_mb_per_slot: int = 4096
    wall_minutes: int = 120
    gpus: int = 0
    span_hosts: int = 1

    @property
    def total_reserved_gb(self) -> float:
        return self.slots * self.mem_mb_per_slot / 1024.0

    def bsub_flags(self) -> List[str]:
        flags = [
            "-n", str(self.slots),
            "-W", str(self.wall_minutes),
            "-M", str(self.mem_mb_per_slot),
            "-R", f"span[hosts={self.span_hosts}]",
            "-R", f"select[mem>={self.mem_mb_per_slot}] "
                  f"rusage[mem={self.mem_mb_per_slot}]",
        ]
        if self.gpus:
            flags += ["-gpu", f"num={self.gpus}/host"]
        return flags


# Parsed out of the LSF epilogue that -oo preserves.
_EPILOGUE = {
    "max_mem_mb": re.compile(r"Max Memory\s*:\s*([\d.]+)\s*(MB|GB)"),
    "requested_mem_mb": re.compile(r"Total Requested Memory\s*:\s*([\d.]+)\s*(MB|GB)"),
    "run_time_s": re.compile(r"Run time\s*:\s*(\d+)\s*sec"),
    "cpu_time_s": re.compile(r"CPU time\s*:\s*([\d.]+)\s*sec"),
    "max_procs": re.compile(r"Max Processes\s*:\s*(\d+)"),
    "max_threads": re.compile(r"Max Threads\s*:\s*(\d+)"),
}


def parse_epilogue(text: str) -> Dict[str, float]:
    """Pull resource usage out of an LSF output file."""
    out: Dict[str, float] = {}
    for key, rx in _EPILOGUE.items():
        m = rx.search(text)
        if not m:
            continue
        val = float(m.group(1))
        if len(m.groups()) > 1 and m.group(2) == "GB":
            val *= 1024
        out[key] = val
    if "cpu_time_s" in out and out.get("run_time_s"):
        out["effective_cores"] = out["cpu_time_s"] / out["run_time_s"]
    return out


def recommend(res: Resources, kind: str = "cpu", n_queues: int = 2) -> List[Queue]:
    """Live-ranked queues to spread this job across."""
    ranked = rank(kind=kind, wall_minutes=res.wall_minutes, slots=res.slots,
                  total_mem_gb=res.total_reserved_gb,
                  span_hosts=res.span_hosts)
    if not ranked:
        raise RuntimeError(
            f"no eligible {kind} queue for wall={res.wall_minutes} min. "
            f"Reduce -W below a short queue's limit or relax the requirement.")
    return ranked[:max(1, n_queues)]


def right_size(res: Resources, epilogue: Dict[str, float],
               headroom: float = 2.0, floor_mb: int = 1024) -> Resources:
    """Return a new Resources sized from a canary's measured peak.

    Memory is set from ``Max Memory`` x ``headroom``, spread over the slots.
    Slot count is cut toward the measured ``effective_cores`` when the job proved
    it cannot use what it reserved -- but never below 1, and never *raised*,
    since a canary cannot show that more parallelism would help.

    Caveat on cutting slots: ``effective_cores = CPU time / Run time`` averages
    over the whole job, so a run with a short parallel phase inside a long serial
    one looks under-parallel even though the parallel phase does use the slots.
    Treat a slot cut as advisory and check it against what the code actually
    does -- e.g. ``prosper.evaluation`` fans bootstrap replicates across a
    ProcessPoolExecutor, so its slots are real when B is large.
    """
    peak = epilogue.get("max_mem_mb")
    slots = res.slots
    eff = epilogue.get("effective_cores")
    if eff is not None and eff + 1 < slots:
        slots = max(1, int(eff + 1))
    mem = res.mem_mb_per_slot
    if peak is not None:
        mem = max(floor_mb, int(peak * headroom / max(1, slots)) + 1)
    return Resources(slots=slots, mem_mb_per_slot=mem,
                     wall_minutes=res.wall_minutes, gpus=res.gpus,
                     span_hosts=res.span_hosts)


def _bsub(cmd: str, *, job_name: str, queue: str, res: Resources,
          log_dir: Path, array: Optional[str] = None,
          dry_run: bool = True, dependency: Optional[str] = None) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"{job_name}[{array}]" if array else job_name
    stem = "%J_%I" if array else "%J"
    argv = ["bsub", "-J", name, "-q", queue,
            # -oo/-eo, not an in-job redirect: keeps the resource epilogue.
            "-oo", str(log_dir / f"{job_name}.{stem}.out"),
            "-eo", str(log_dir / f"{job_name}.{stem}.err"),
            *res.bsub_flags()]
    if dependency:
        # Chain on `ended(...)` by preference (standing rule 8): `done()` on an
        # array becomes unsatisfiable when one element fails and LSF then
        # kills the whole downstream chain with no log.
        argv += ["-w", dependency]
    argv.append(cmd)
    printable = " ".join(shlex.quote(a) for a in argv)
    if dry_run:
        print(printable)
        return ""
    env = os.environ.copy()
    env["LSB_JOB_REPORT_MAIL"] = "N"
    out = subprocess.run(argv, capture_output=True, text=True, check=True,
                         env=env).stdout
    m = re.search(r"Job <(\d+)>", out)
    print(out.strip())
    return m.group(1) if m else ""


def one(cmd: str, *, job_name: str, res: Resources, log_dir: Path,
        kind: str = "cpu", queue: Optional[str] = None,
        dry_run: bool = True, dependency: Optional[str] = None) -> str:
    """Submit a single non-array job to the least-loaded eligible queue.

    Used for the gates, which are a handful of discrete jobs rather than a
    sweep. Pass ``queue=`` to override the live ranking.
    """
    if queue is None:
        q = recommend(res, kind=kind, n_queues=1)[0]
        queue = q.name
        where = (f"queue={q.name} pend={q.pend} run={q.run} "
                 f"pressure={q.pressure:.2f}")
    else:
        where = f"queue={queue} (explicit)"
    print(f"[submit] {job_name}: {where}, reserving "
          f"{res.total_reserved_gb:.1f} GB total over {res.slots} slot(s)"
          f"{f', after {dependency}' if dependency else ''}")
    return _bsub(cmd, job_name=job_name, queue=queue, res=res,
                 log_dir=log_dir, dry_run=dry_run, dependency=dependency)


def canary(cmd: str, *, job_name: str, res: Resources, log_dir: Path,
           kind: str = "cpu", dry_run: bool = True) -> str:
    """Submit a single element to the least-loaded eligible queue.

    Run this, wait for it, then feed :func:`parse_epilogue` output to
    :func:`right_size` before calling :func:`fan_out`.
    """
    return one(cmd, job_name=f"{job_name}_canary", res=res, log_dir=log_dir,
               kind=kind, dry_run=dry_run)


def fan_out(cmd: str, *, n: int, job_name: str, res: Resources, log_dir: Path,
            kind: str = "cpu", n_queues: int = 2,
            canary_epilogue: Optional[Dict[str, float]] = None,
            dry_run: bool = True, dependency: Optional[str] = None,
            queues: Optional[List[str]] = None) -> List[str]:
    """Submit ``1..n`` as arrays split across the least-loaded eligible queues.

    ``cmd`` should reference ``$LSB_JOBINDEX``. Refuses to run without a canary
    epilogue unless explicitly dry-run, because that is the whole point.
    """
    if canary_epilogue is None and not dry_run:
        raise RuntimeError(
            "fan_out requires canary_epilogue: submit one element first, read its "
            "epilogue, then size the sweep. Pass dry_run=True to preview.")
    if canary_epilogue:
        res = right_size(res, canary_epilogue)
        print(f"[fan_out] right-sized to -n {res.slots} -M {res.mem_mb_per_slot} "
              f"({res.total_reserved_gb:.1f} GB total reserved)")

    ranges = chunk_array(n, MAX_JOB_ARRAY_SIZE)
    if queues:
        qnames = list(queues)
    else:
        qnames = [q.name for q in recommend(res, kind=kind, n_queues=n_queues)]
    print(f"[fan_out] {n} elements -> {len(ranges)} legal array(s) "
          f"(cap {MAX_JOB_ARRAY_SIZE}) across {qnames}")

    # Split each legal array range across the chosen queues so no single queue
    # holds the whole sweep.
    job_ids: List[str] = []
    for lo, hi in ranges:
        total = hi - lo + 1
        per = max(1, total // len(qnames))
        start = lo
        for i, qn in enumerate(qnames):
            end = hi if i == len(qnames) - 1 else min(hi, start + per - 1)
            if start > end:
                break
            job_ids.append(_bsub(cmd, job_name=job_name, queue=qn, res=res,
                                 log_dir=log_dir, array=f"{start}-{end}",
                                 dry_run=dry_run, dependency=dependency))
            start = end + 1
    return job_ids


def stalled(job_ids: List[str], minutes: int = 15) -> List[str]:
    """Job IDs still PEND after ``minutes``, with LSF's pending reason.

    Use this to time-box a sweep: if a chunk has not started, resubmit the
    remainder to the next queue rather than waiting.
    """
    out = []
    for jid in job_ids:
        if not jid:
            continue
        r = subprocess.run(["bjobs", "-o", "stat", "-noheader", jid],
                           capture_output=True, text=True)
        if r.stdout.strip() == "PEND":
            why = subprocess.run(["bjobs", "-p", jid], capture_output=True,
                                 text=True).stdout.strip()
            out.append(f"{jid}: {why}")
    return out
