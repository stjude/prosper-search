"""LSF queue inventory and live-pressure probe.

Why this module exists
----------------------
Every existing driver in the ``comet-*`` trees submits to ``standard`` (or to no
queue at all, taking the LSF default, which *is* ``standard``). Measured on this
cluster 2026-08-27:

    standard          PRIO  50   PEND 32,154   RUN 9,390    hosts rhel8_cpu
    priority          PRIO 100   PEND    108   RUN   565    hosts rhel8_cpu
    large_core_count  PRIO  60   PEND      0   RUN   320    hosts large_core
    large_mem         PRIO  60   PEND      0   RUN   479    hosts rhel8_lm
    short             PRIO 100   PEND      2   RUN     0    hosts rhel8_cpu (30 min wall)
    heavy_io          PRIO  50   PEND      0   RUN   170    JL/U 500, good for I/O-bound array work

    gpu               PRIO  80   PEND 26,456   RUN 1,027
    gpu_priority      PRIO 100   PEND    420   RUN   256
    gpu_short         PRIO 100   PEND      0   RUN     0            (30 min wall)

``priority`` and ``standard`` draw from the *same host group* (``rhel8_cpu``) and
both are ``USERS: all``, so submitting a CPU sweep to ``standard`` puts it behind
32,000 jobs for no reason. A 128-job sweep landing on ``standard`` is the failure
mode this module exists to prevent: it is not slow, it is indefinitely queued.

Numbers here are a *starting prior*. Always call :func:`probe` -- queue pressure
moves by tens of thousands of jobs within a day.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# LSF cluster limit. A [1-4307] array is rejected outright, not truncated.
MAX_JOB_ARRAY_SIZE = 4000


# An idle queue with fewer than this many pending jobs is quiet, not stalled.
STALL_PEND = 10

# At or above this slot count with span[hosts=1], placement rather than
# contention becomes the binding constraint (measured 2026-08-27, see
# Queue.pressure). Empirically -n 10 placed fine on short/priority; -n 32 did
# not, and only large_core_count took it.
WIDE_JOB_SLOTS = 16

# Queues whose hosts are actually provisioned for wide single-host jobs.
WIDE_JOB_QUEUES = ("large_core_count", "large_mem")

# Queues that rank() may report but must never select (standing rule 7).
NEVER_RANK = ("standard",)

# LSF pending reasons that mean "this queue will never place this shape", as
# opposed to "wait your turn". Matched case-insensitively as substrings.
UNPLACEABLE_REASONS = (
    "affinity resource requirement cannot be met",
    "cannot be met",
    "job requirements for reserving resource",
    "not enough slots or resources",
)


@dataclass
class Queue:
    name: str
    prio: int = 0
    pend: int = 0
    run: int = 0
    kind: str = "cpu"                  # "cpu" | "gpu"
    wall_limit_min: Optional[int] = None   # None = no limit
    host_group: str = ""
    note: str = ""

    # Placement is a separate question from contention, and conflating them cost
    # a real submission cycle on 2026-08-27: `priority` and `heavy_io` both read
    # PEND 0 / RUN >0 -- the best possible pressure -- and then both PEND'd
    # immediately on `-n 32 -R span[hosts=1]` with "Affinity resource
    # requirement cannot be met", while `large_core_count` placed the identical
    # job within seconds.
    #
    # It is not a core-count limit: 233 hosts on this cluster have 64 cores and
    # 56 have 128. It is CPUSET_CPU_EXCLUSIVE affinity against current
    # occupancy, which no queue-level counter exposes. So:
    #
    #   pressure ranks CONTENTION. It does not predict PLACEMENT for a shape.
    #
    # Consequence for how these functions must be used: for wide single-host
    # jobs (>= WIDE_JOB_SLOTS with span[hosts=1]), prefer a queue whose hosts
    # are provisioned for it, and treat an affinity pending reason as a signal
    # to move on rather than to wait. See `pending_reason_kind`.
    @property
    def pressure(self) -> float:
        """Pending per running slot. Lower is better; ``inf`` means nothing is
        draining. This is the ranking signal, not PRIO -- a high-priority queue
        with 26,000 pending jobs is still a wall.

        The ``run == 0`` case needs care, and the naive reading of it is wrong.
        An idle queue holding a couple of pending jobs is *quiet*, which is the
        best thing a queue can be; only an idle queue holding a real backlog is
        stalled. Treating both as ``inf`` ranked ``short`` -- reliably the
        emptiest CPU queue here -- dead last, which is precisely backwards.
        """
        if self.run <= 0:
            return float("inf") if self.pend > STALL_PEND else 0.0
        return self.pend / float(self.run)


# Candidate queues, in preference order, for work we actually submit. Restricted
# or foreign queues (cryoem, dragen, biostat, ...) are deliberately absent.
CPU_CANDIDATES: List[Queue] = [
    Queue("priority",         prio=100, kind="cpu", host_group="rhel8_cpu",
          note="same hosts as standard, PRIO 100, USERS all -- the default choice"),
    Queue("heavy_io",         prio=50,  kind="cpu", host_group="",
          note="JL/U 500; good fit for I/O-bound work such as WSI header reads"),
    Queue("large_core_count", prio=60,  kind="cpu", host_group="large_core",
          note="for -n >= 32; historically zero pending"),
    Queue("standard",         prio=50,  kind="cpu", host_group="rhel8_cpu",
          note="the cluster default; usually the most backed-up queue on it"),
    Queue("short",            prio=100, kind="cpu", host_group="rhel8_cpu",
          wall_limit_min=30,
          note="30 min hard wall; near-empty, so ideal for per-slide array elements"),
    Queue("large_mem",        prio=60,  kind="cpu", host_group="rhel8_lm",
          note="only for genuinely >500 GB jobs; do not use to dodge a queue"),
]

GPU_CANDIDATES: List[Queue] = [
    Queue("gpu_short",    prio=100, kind="gpu", wall_limit_min=30,
          note="30 min wall; frequently completely empty"),
    Queue("gpu_priority", prio=100, kind="gpu",
          note="dgx_80g + priority reserve; per-user JL/U 128"),
    Queue("rhel88_gpu",   prio=20,  kind="gpu",
          note="bqueues USERS says devneuro_gputest, but a real job log shows "
               "user executing here; PRIO 20 quiet pocket"),
    # dgx and rhel9_test_gpu added 2026-08-27 after probing every GPU queue on
    # the cluster with a real `bsub` rather than reading `bqueues USERS`, which
    # is not the effective gate here (rhel88_gpu's USERS field lists a group
    # this user is not in, yet accepts the job). `dgx` is the find: PRIO 100
    # and 0 pending against gpu_priority's 420.
    Queue("dgx",          prio=100, kind="gpu",
          note="PRIO 100, historically 0 pending; probe-confirmed usable"),
    # rhel9_test_gpu REMOVED 2026-08-27 after it accepted 16 array elements and
    # lost all of them: 8 EXIT + 8 UNKWN on its single host nodegpu216. torch
    # and the encoder loaded fine, then every DataLoader worker died with
    # ModuleNotFoundError -- the RHEL9 host cannot import the RHEL8-built
    # openslide extension that this miniconda interpreter links against. So the
    # queue is submittable but not *runnable* for this pipeline, which is a
    # worse failure mode than a rejection: `bsub` succeeds and the work
    # silently evaporates. Any queue outside the rhel8 host groups needs an
    # end-to-end canary, not just a submit probe.
    # gpu_rtx REMOVED from the candidate list 2026-08-27: `bsub -q gpu_rtx`
    # returns "User cannot use the queue. Job not submitted." for user, so
    # ranking it wasted a fan_out leg and aborted a submission mid-flight.
    # Same measured result for kellogg_gpu, biostat_gpu, cryoem_rtx,
    # cryo_core and hpcf_test_gpu -- all rejected, all deliberately absent.
    Queue("gpu",          prio=80,  kind="gpu",
          note="the obvious choice and usually the worst; check PEND first"),
]


def available() -> bool:
    return shutil.which("bqueues") is not None


def probe(names: Optional[List[str]] = None) -> Dict[str, Queue]:
    """Read live PEND/RUN/PRIO from ``bqueues``.

    Returns a dict keyed by queue name. Unknown or closed queues are omitted.
    """
    if not available():
        raise RuntimeError("bqueues not on PATH; run this on a submit host")
    out = subprocess.run(
        ["bqueues", "-o", "QUEUE_NAME:24 STATUS:14 PRIO:8 PEND:10 RUN:10"],
        capture_output=True, text=True, check=True).stdout

    known = {q.name: q for q in CPU_CANDIDATES + GPU_CANDIDATES}
    live: Dict[str, Queue] = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        name, status, prio, pend, run = parts[0], parts[1], parts[2], parts[-2], parts[-1]
        if names and name not in names:
            continue
        if not status.startswith("Open"):
            continue
        base = known.get(name)
        try:
            q = Queue(
                name=name,
                prio=int(prio),
                pend=int(pend),
                run=int(run),
                kind=base.kind if base else "cpu",
                wall_limit_min=base.wall_limit_min if base else None,
                host_group=base.host_group if base else "",
                note=base.note if base else "",
            )
        except ValueError:
            continue
        live[name] = q
    return live


def rank(kind: str = "cpu", wall_minutes: Optional[int] = None,
         slots: int = 1, total_mem_gb: Optional[float] = None,
         span_hosts: int = 1) -> List[Queue]:
    """Live-ranked eligible queues, best first.

    Eligibility filters on the queue's wall limit and, for wide jobs, steers
    ``-n >= 32`` toward ``large_core_count``. Ranking is by pressure
    (pending/running), then by descending PRIO.
    """
    candidates = CPU_CANDIDATES if kind == "cpu" else GPU_CANDIDATES
    live = probe([q.name for q in candidates])
    eligible: List[Queue] = []
    for base in candidates:
        q = live.get(base.name)
        if q is None:
            continue
        if base.name in NEVER_RANK:
            # Standing rule 7: never `standard`. It stayed in the candidate list
            # for reporting, and on 2026-08-29 the live ranking picked it for a
            # retry array once this user's own 2,887-element fan-out had pushed
            # priority/heavy_io pressure above standard's. Reportable, never
            # selectable.
            continue
        if q.wall_limit_min is not None and wall_minutes is not None \
                and wall_minutes > q.wall_limit_min:
            continue
        if base.name == "large_mem" and (
                slots < 32 or (total_mem_gb is not None and total_mem_gb < 500)):
            continue          # large_mem is for >500 GB jobs, not for queue-dodging
        if base.name == "large_core_count" and slots < 32:
            continue          # likewise: it exists for high-core-count jobs
        eligible.append(q)
    # For wide single-host jobs, placement dominates: a zero-pressure queue that
    # cannot satisfy the affinity requirement is worse than a busier one that
    # can. So wide-job queues sort first, and pressure only orders within each
    # group. Narrow jobs are unaffected.
    wide = span_hosts == 1 and slots >= WIDE_JOB_SLOTS
    eligible.sort(key=lambda q: (
        0 if (wide and q.name in WIDE_JOB_QUEUES) else 1,
        q.pressure, -q.prio))
    return eligible


def pending_reason_kind(job_id: str) -> str:
    """Classify why a job is pending: ``"unplaceable"``, ``"queued"`` or ``"unknown"``.

    ``"unplaceable"`` means the queue cannot satisfy the resource shape at all,
    so waiting is pointless and the right move is to resubmit elsewhere
    immediately. That distinction is invisible from PEND/RUN counts.
    """
    try:
        out = subprocess.run(["bjobs", "-p", str(job_id)],
                             capture_output=True, text=True).stdout.lower()
    except Exception:
        return "unknown"
    if not out.strip():
        return "unknown"
    for r in UNPLACEABLE_REASONS:
        if r in out:
            return "unplaceable"
    return "queued"


def report(kind: str = "cpu", wall_minutes: Optional[int] = None,
           slots: int = 1, total_mem_gb: Optional[float] = None) -> str:
    rows = rank(kind=kind, wall_minutes=wall_minutes, slots=slots,
                total_mem_gb=total_mem_gb)
    if not rows:
        return f"no eligible {kind} queue for wall={wall_minutes} min, slots={slots}"
    w = max(len(q.name) for q in rows)
    lines = [f"{'QUEUE':<{w}}  {'PRIO':>4}  {'PEND':>7}  {'RUN':>6}  {'PRESSURE':>8}  NOTE"]
    for q in rows:
        p = "inf" if q.pressure == float("inf") else f"{q.pressure:.2f}"
        lines.append(f"{q.name:<{w}}  {q.prio:>4}  {q.pend:>7}  {q.run:>6}  {p:>8}  {q.note}")
    return "\n".join(lines)


def chunk_array(n: int, max_size: int = MAX_JOB_ARRAY_SIZE) -> List[tuple]:
    """Split ``1..n`` into LSF-legal array ranges.

    ``MAX_JOB_ARRAY_SIZE`` is 4000 here, so a naive ``[1-4307]`` array for the
    4,307-slide CCDI cohort is rejected by the scheduler rather than truncated.
    """
    return [(s + 1, min(s + max_size, n)) for s in range(0, n, max_size)]
