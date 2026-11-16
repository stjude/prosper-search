#!/usr/bin/env python3
"""R2.1 route (b) UNI2 launcher (specification candidate 002, adapter A-3).

Modelled on the qf section of ``submit_r2_1_filter.py`` (unchanged, frozen). Every stage only PRINTS
its exact bsub lines unless ``--go`` is given; with ``--go`` it submits HELD (``bsub -H``), appends
every job id to ``ROOT/launcher/journal.jsonl`` and leaves release to ``readback`` + ``release``.

Queue rule (specification section 8, PROBE_RECORD.md 2026-10-13):
  * ``queues``        enumerate ``bqueues -w`` (saved under ROOT/launcher/bqueues/)
  * ``probe``         one short probe per candidate queue (-M/-W = the wave's largest element),
                      outputs under ROOT/queue_probe/<wave>/
  * ``probe-status``  writes ROOT/queue_probe/<wave>/dispatching.json from the probes' bjobs states
  * production stages split their elements across the queues that dispatched in that wave
    (default: the six RHEL 8.10 production queues; RHEL 9 only with ``--allow-rhel9 <bitwise
    fidelity record>``; ``short`` only for stages with -W <= 30)
  * ``migrate``       bswitch still-PEND elements of this package's own journaled jobs
Every bsub carries -o/-e files and runs with LSB_JOB_REPORT_MAIL=N; -N/-u are never used.
Job 326997606 and any job named *_DD_R1_* are never touched.

Repair candidate 002 (implementation review 001, LEAD_IMPL_DECISIONS_001):
  * release only after a PASS readback whose job-ID set equals the wave's unreleased set (B1);
    the readback parses every ``bjobs -l`` element block against the journaled packet (N2)
  * every --go submission is once-only, keyed on the journal (B2); gap-fill is the only exception
    and is itself once per element across reasons (B3)
  * every job exports the six *_NUM_THREADS=1, PYTHONHASHSEED=0 and LSB_JOB_REPORT_MAIL=N and records
    them in its host record (B4)
  * dry runs write only under a scratch directory outside ROOT (N5)
  * ``fidelity-check`` / ``--requires`` refuse fidelity markers that record dev/test/pin-override use or
    whose code/pins/outputs do not match (N6); ``collect`` writes ROOT/launcher/provenance_sums (N7)

Stages: queues, probe, probe-status, qf-features, qf-views, qf-sim, f2-rows, combine-qf,
evaluate-qf, single, readback, release, migrate, status, collect, fidelity-check.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PS))
from prosper.evaluation import r2_1_filter_eval_cli as EV   # noqa: E402  (frozen; names only)

HAORAN = Path("/PROSPER_WORKSPACE")
OUT = HAORAN / "prosper_revision" / "prosper_revision_output"
ROOT_NAME = "r2_1_route_b_uni2_20261013"
ROOT = OUT / ROOT_NAME
WORKER = PS / "scripts" / "lsf" / "prosper_job.sh"
FEATURES_DIR = HAORAN / "comet-feature-gen" / "comet-uni2" / "data" / "tmp_final"
PYTHON = "/HOME/miniconda3/bin/python3"
ENCODER, FEATURE_DIM = "uni2", 1536
BUDGETS, FOLDS = (5, 100), (0, 1, 2, 3, 4)
ARMS = {"both": "both_filtered", "query": "query_only"}
METHODS = ("ot", "bob")
BATCH = {"ot": 1, "bob": 25}            # only the bob25 parametrisation is mirrored
UNFILTERED_VIEW = "library_unfiltered_pop2604"
POPULATION_SHA = "c7cae1f0d855952e84e85aa90c7ef7d8a622086273c113285655c96be946d97b"
PUBLISHED_DIR = HAORAN / "comet-feature-gen" / "comet-uni2" / "output" / "similarity_matrix"
ALL_TISSUE = {
    "ot_cosine": (PUBLISHED_DIR / "combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl",
                  "a288d8926c8715c86ae3063ff68d9f7b0c6b365eaa77549f45aa02142d2e1714"),
    "yottixel": (PUBLISHED_DIR / "combined_similarity_matrix_uni2_yottixel_no_same_patient.pkl",
                 "95b4bd13c4859c3e4460c4045f2a6b18a9a66cc894c070540f5ccc6dda1283e5"),
}
F2_SLIDES = ("11465", "5541", "16727", "5506", "12129", "11530", "6215", "7862", "7449", "11513")
F2_OT_SLIDES = ("11465", "16727", "12129", "6215", "7449")   # first listed slide per fold (reference)

RHEL8_QUEUES = ("standard", "priority", "heavy_io", "large_mem", "large_core_count", "superdome")
RHEL9_QUEUES = ("rhel9_standard", "rhel9_large_mem")
SHORT_QUEUE, SHORT_LIMIT_MIN = "short", 30
PROTECTED_JOB_IDS = frozenset({"326997606"})
PROTECTED_NAME_PATTERN = "*_DD_R1_*"
NATIVE_ONE = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
MAX_ARRAY = 4000
DONE_STATES, ACTIVE_STATES = {"DONE"}, {"PEND", "PSUSP", "RUN", "USUSP", "SSUSP", "WAIT"}
PROBE_MAX_AGE_HOURS = 24                     # N4: production uses only current-wave probes <= 24 h old
SCRATCH_PARENT = None                        # N5: parent of dry-run scratch dirs (None = tempfile default)
_SCRATCH: Path | None = None                 # set by main() for dry runs; never under ROOT

# Resources (MB, minutes) mirror the reference launcher / specification section 8.
RES = {
    ("sim", "both", "ot"): (28672, 600), ("sim", "query", "ot"): (32768, 900),
    ("sim", "both", "bob"): (10240, 600), ("sim", "query", "bob"): (10240, 600),
    "features": (8192, 120), "views": (4096, 120), "combine": (8192, 60), "evaluate": (16384, 120),
    "f2_ot": (32768, 30),       # specification section 8: 5 OT rows, 32 GiB / 30 min
    "f2_bob": (10240, 600),     # specification section 8 BoB shape (1 CPU / 10 GiB / -W 600); review 001 #23
}
# Specification section 8 table (candidate 002). Views (4096/120) is not stated there ("+ 1 views job").
CONTRACT_S8 = {
    ("sim", "both", "ot"): (28672, 600), ("sim", "query", "ot"): (32768, 900),
    ("sim", "both", "bob"): (10240, 600), ("sim", "query", "bob"): (10240, 600),
    "features": (8192, 120), "combine": (8192, 60), "evaluate": (16384, 120),
    "f2_ot": (32768, 30), "f2_bob": (10240, 600),
}
if any(RES[k] != v for k, v in CONTRACT_S8.items()) or BATCH["bob"] != 25:
    raise SystemExit("[route-b launcher] STOP: RES/BATCH differ from the contract section 8 table")

# N6: the accepted fidelity code (implementation 001 + g5_replay.py of candidate 002).
FIDELITY_SCRIPTS = OUT / ROOT_NAME / "fidelity" / "scripts"
FIDELITY_CODE_SHA = {
    "rbfid.py": "30be23b6b854f89f709d10ac3458c9826b6f3d8acf885e9124f4ee8014523748",
    "f2_compare.py": "1fa81f3b7e0f0bd9869f457862ebf51856a06d499a407f81cb793d067704c553",
    "f2_compare_a001.py": "c66209b2b7b628cc60abf9691d083a633a7f95b095e3d35f2144ce063d98d2c2",
    "f2_select.py": "64731b6f948e2e30f43477890dd7a09b8a145114f8ef7f39ffd02d1dedd9c543",
    "f3_banks.py": "a938964d8dc4f6d47c2993e0f901f88251c284432a78a299d5b8f4566c993c17",
    "f4_join.py": "316c4679417dfbe0af38ba2bc31dfe00b03a0c49dd3da671d162a6bd72ba0705",
    "g3_head_hash_inequality.py": "667e808c57fc4d09451fd014fce527d1722ae3566ab51f4c4fabb76af5ff7b68",
    "g5_replay.py": "ff648ecd39bfd7408a4580b7d40516130e422d6c39d0fb2d722f05b681700447",
    "g7_rows.py": "dc8b47d493bdcbcf6e3735154ff8be4824adddcc140d2fdd2b479b7954c38935",
    "m1_recheck.py": "1d60840c4c0ac10708c9e865572fc0f538d6cef229f9c6a4c493af6548a98f74",
    "near_tie_census.py": "28bd944daf349789dfdef006d7f333abe1c55e7bb5ac4d971c9466d744eb0095",
    "s0_census.py": "eb724dee3c268d4aebd4b6171cc831832a3a55dfe4cd81592c4687c62cf25e36",
}
RHEL9_OS = re.compile(r"^Red Hat Enterprise Linux 9(\.\d+)?\b")


class LaunchError(SystemExit):
    pass


def die(msg: str):
    raise LaunchError(f"[route-b launcher] STOP: {msg}")


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def launcher_sha() -> str:
    return sha256_file(Path(__file__))


def parse_ts(text: str) -> _dt.datetime:
    t = _dt.datetime.fromisoformat(str(text))
    return t if t.tzinfo else t.astimezone()


def write_new(path: Path, text: str):
    """Refuse-to-change write: create, or accept identical existing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            die(f"{path} exists with different content; refusing to overwrite")
        return
    with open(path, "x") as fh:
        fh.write(text)


# ------------------------------------------------------------------ LSF plumbing / guards
def run_cmd(argv: list, env=None) -> str:
    """Single choke point for every LSF command (monkeypatched in tests)."""
    return subprocess.run(argv, capture_output=True, text=True, check=True, env=env).stdout


def lsf_env() -> dict:
    env = os.environ.copy()
    env["LSB_JOB_REPORT_MAIL"] = "N"
    env["PROSPER_OUT"] = str(OUT)
    return env


def protected(job_id: str, name: str = "") -> bool:
    base = str(job_id).split("[")[0]
    return base in PROTECTED_JOB_IDS or fnmatch.fnmatchcase(name or "", PROTECTED_NAME_PATTERN)


def guard_own(job_id: str, name: str = ""):
    base = str(job_id).split("[")[0]
    if protected(base, name):
        die(f"refusing to touch protected job {job_id} ({name})")
    if base not in journal_ids():
        die(f"job {job_id} is not in this package's journal; refusing to touch it")


def bsub_argv(*, name: str, queue: str, mem_mb: int, wall_min: int, out: Path, err: Path, cmd: str,
              array: str | None = None, dependency: str | None = None, hold: bool = True) -> list:
    if protected("", name):
        die(f"job name {name} matches {PROTECTED_NAME_PATTERN}")
    argv = ["bsub", "-J", f"{name}[{array}]" if array else name, "-q", queue,
            "-o", str(out), "-e", str(err),
            "-n", "1", "-W", str(wall_min), "-M", str(mem_mb),
            "-R", "span[hosts=1]", "-R", f"select[mem>={mem_mb}] rusage[mem={mem_mb}]"]
    if hold:
        argv.append("-H")
    if dependency:
        argv += ["-w", dependency]
    argv.append(cmd)
    if any(a in ("-N", "-u", "-oo", "-eo") for a in argv[:-1]):
        die("bsub argv carries a mail/overwrite flag")
    return argv


def dep(ids) -> str | None:
    ids = [i for i in ids if i]
    return " && ".join(f"ended({i})" for i in ids) or None


def env_exports() -> str:
    """B4: exported first in EVERY job command (worker stages, single .py, probes)."""
    return "export " + " ".join([*(f"{v}=1" for v in NATIVE_ONE), "PYTHONHASHSEED=0", "LSB_JOB_REPORT_MAIL=N",
                                 f"PROSPER_OUT={shlex.quote(str(OUT))}"])


ENV_RECORD_PATTERN = "^(" + "|".join(NATIVE_ONE) + "|PYTHONHASHSEED|LSB_JOB_REPORT_MAIL|PROSPER_OUT)="


def host_preamble(stage: str) -> str:
    """Addendum A4b: host, OS release and CPU model per element (written in-job); B4: the job's thread /
    hash-seed / mail environment is recorded in the same file."""
    d = ROOT / "host_records" / stage
    return (f"mkdir -p {shlex.quote(str(d))} && {{ echo \"host=$(hostname)\"; "
            f"grep '^PRETTY_NAME=' /etc/os-release; grep -m1 'model name' /proc/cpuinfo; "
            f"echo \"lsb_jobid=$LSB_JOBID lsb_jobindex=${{LSB_JOBINDEX:-0}} lsb_queue=$LSB_QUEUE\"; "
            f"env | grep -E '{ENV_RECORD_PATTERN}' | sort | sed 's/^/env:/'; "
            f"date -Is; }} > {shlex.quote(str(d))}/${{LSB_JOBID}}_${{LSB_JOBINDEX:-0}}.txt")


def sha_check(pairs) -> str:
    """In-job re-hash of pinned inputs (STOP on mismatch: sha256sum -c exits non-zero)."""
    lines = "\\n".join(f"{sha}  {path}" for path, sha in pairs)
    return f"printf '{lines}\\n' | sha256sum -c --strict --quiet -"


def worker_cmd(stage: str, module: str, args: str = "", params: Path | None = None, pre=(), post=()) -> str:
    env = [f"PROSPER_MODULE={module}", f"PROSPER_ROOT={PS}", f"PROSPER_OUT={OUT}",
           f"PROSPER_PYTHON={PYTHON}", "LSB_JOB_REPORT_MAIL=N"]
    if params:
        env.append(f"PROSPER_PARAMS={params}")
    parts = [env_exports(), host_preamble(stage), *pre,
             " ".join(env + ["bash", str(WORKER)]) + (f" {args}" if args else "")]
    return " && ".join([*parts, *post])


# ------------------------------------------------------------------ journal
def journal_path() -> Path:
    return ROOT / "launcher" / "journal.jsonl"


def journal_records() -> list:
    p = journal_path()
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def journal_ids() -> set:
    return {str(r["job_id"]) for r in journal_records() if r.get("job_id")}


def journal_append(rec: dict):
    p = journal_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps({"ts": now(), **rec}, sort_keys=True) + "\n")


def expand_spec(spec) -> set:
    """'3-5,9' -> {3, 4, 5, 9}."""
    out = set()
    for part in str(spec).split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out |= set(range(int(lo), int(hi) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def record_indices(r: dict):
    """Element indices a journaled submit covers (None = a non-array job)."""
    if r.get("gap_indices"):
        return set(int(i) for i in r["gap_indices"])
    return expand_spec(r["array"]) if r.get("array") else None


def once_guard(name: str, array, params, gap_fill: bool):
    """B2: a --go submission is refused when the journal already holds a submit of the same job name or the
    same params file whose element indices overlap (or either is a non-array job). Only the explicit gap-fill
    path is exempt (it has its own once-per-element rule, B3)."""
    if gap_fill:
        return
    new = expand_spec(array) if array else None
    for r in journal_records():
        if r.get("action") != "submit":
            continue
        if r.get("job_name") != name and not (params and r.get("params") == str(params)):
            continue
        old = record_indices(r)
        if new is None or old is None or new & old:
            die(f"once-guard: {name} {('[' + array + ']') if array else ''} overlaps journaled job {r['job_id']} "
                f"({r.get('job_name')} {r.get('array') or ''} {r.get('params') or ''}); every element is "
                f"submitted once (contract 8.7/12.8); a failed element is only resubmitted via --gap-fill")


class Submitter:
    def __init__(self, go: bool, wave: str, stage: str):
        self.go, self.wave, self.stage = go, wave, stage
        self.lines, self.job_ids, self.requires = [], [], []

    def submit(self, *, name, queue, mem_mb, wall_min, log_dir: Path, cmd, array=None, dependency=None,
               extra=None, label: str | None = None) -> str:
        stem = "%J_%I" if array else "%J"
        argv = bsub_argv(name=name, queue=queue, mem_mb=mem_mb, wall_min=wall_min,
                         out=log_dir / f"{name}.{stem}.out", err=log_dir / f"{name}.{stem}.err",
                         cmd=cmd, array=array, dependency=dependency, hold=True)
        printable = shlex.join(argv)
        self.lines.append(printable)
        print(printable)
        if not self.go:
            return f"<{name}{'[' + array + ']' if array else ''}>"
        extra = dict(extra or {})
        once_guard(name, array, extra.get("params"), bool(extra.get("gap_fill")))
        if "params" in extra:
            extra["params_sha256"] = sha256_file(Path(extra["params"]))
        log_dir.mkdir(parents=True, exist_ok=True)
        out = run_cmd(argv, env=lsf_env())
        m = re.search(r"Job <(\d+)>", out)
        if not m:
            die(f"bsub returned no job id: {out!r}")
        jid = m.group(1)
        journal_append({"action": "submit", "stage": self.stage, "wave": self.wave, "job_id": jid,
                        "job_name": name, "queue": queue, "array": array, "mem_mb": mem_mb,
                        "wall_min": wall_min, "dependency": dependency, "held": True, "argv": argv,
                        "cmd_sha256": sha256_text(cmd), "launcher_sha256": launcher_sha(),
                        "label": label or name, "log_dir": str(log_dir), "submit_env_mail": "N", **extra})
        self.job_ids.append(jid)
        return jid

    def finish(self):
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")
        if not self.go:
            d = scratch() / "launcher" / "dryrun"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{self.stage}_{self.wave or 'nowave'}_{stamp}.txt").write_text("\n".join(self.lines) + "\n")
            print(f"[dry-run] listing and dry-run params under {scratch()} (nothing written under ROOT)")
            return
        d = ROOT / "launcher" / "submitted"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self.stage}_{self.wave or 'nowave'}_{stamp}.txt").write_text("\n".join(self.lines) + "\n")
        write_input_bindings(self, stamp)


def git_diagnostics() -> dict:
    """Review 001 #1 (analyst ruling): git HEAD and porcelain status of PS are DIAGNOSTICS; hash pins govern."""
    res = {}
    for key, argv in (("git_head", ["git", "-C", str(PS), "rev-parse", "HEAD"]),
                      ("git_status_porcelain", ["git", "-C", str(PS), "status", "--porcelain"])):
        try:
            res[key] = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=120).stdout
        except Exception as exc:   # noqa: BLE001  (diagnostic only)
            res[key] = f"<unavailable: {exc!r}>"
    res["git_head"] = res["git_head"].strip()
    return res


def write_input_bindings(sub: Submitter, stamp: str):
    rec = {"created": now(), "stage": sub.stage, "wave": sub.wave, "job_ids": sub.job_ids,
           "launcher": str(Path(__file__).resolve()), "launcher_sha256": launcher_sha(),
           "worker": str(WORKER), "worker_sha256": sha256_file(WORKER),
           "required_fidelity_markers": sub.requires,
           "diagnostics_not_pins": git_diagnostics()}
    d = ROOT / "launcher" / "input_bindings"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{sub.stage}_{sub.wave or 'nowave'}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")


def scratch() -> Path:
    """N5: dry-run write root (outside ROOT)."""
    if _SCRATCH is None:
        die("internal: dry-run scratch directory not initialised")
    return _SCRATCH


def write_base() -> Path:
    """ROOT for --go, the scratch directory for dry runs."""
    return ROOT if _SCRATCH is None else _SCRATCH


def init_scratch(path) -> Path:
    rr = ROOT.resolve()
    for cand in [x for x in (path, SCRATCH_PARENT) if x is not None]:
        rp = Path(cand).resolve()
        if rp == rr or rr in rp.parents:
            die(f"scratch {cand} lies under ROOT {ROOT}; dry runs never write under ROOT (N5)")
    if path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
    else:
        if SCRATCH_PARENT is not None:
            Path(SCRATCH_PARENT).mkdir(parents=True, exist_ok=True)
        p = Path(tempfile.mkdtemp(prefix="r21rb_dryrun_", dir=SCRATCH_PARENT))
    rp = p.resolve()
    if rp == rr or rr in rp.parents:
        die(f"scratch {p} lies under ROOT {ROOT}; dry runs never write under ROOT (N5)")
    return p


# ------------------------------------------------------------------ queues
def enumerate_queues(save: bool = True) -> dict:
    out = run_cmd(["bqueues", "-w"])
    if save:
        d = write_base() / "launcher" / "bqueues"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"bqueues_{_dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}.txt").write_text(out)
    status = {}
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) >= 3:
            status[f[0]] = f[2]
    return status


_RHEL9_OK: dict = {}


def validate_rhel9_record(record: Path, sha: str | None) -> dict:
    """N3: the record must be a g7_rows marker (bitwise-PASS rows for OT and BoB) produced on RHEL 9, whose
    sha256 the launcher recomputes and compares with the independently supplied --allow-rhel9-sha. Accepted: COMPLETE.json
    (outcome PASS), or FAILED.json whose only failure is the --only restricted scope (PASS_PARTIAL_SCOPE)."""
    record = Path(record)
    key = (str(record), sha)
    if key in _RHEL9_OK:
        return _RHEL9_OK[key]
    if not record.is_file():
        die(f"--allow-rhel9 record {record} does not exist")
    if not sha:
        die("--allow-rhel9 requires --allow-rhel9-sha <sha256 supplied by the lead>")
    got = sha256_file(record)
    if got != sha:
        die(f"--allow-rhel9 record sha256 {got} != --allow-rhel9-sha {sha}")
    chk = check_fidelity_marker(record, allow_partial_scope=True)
    m = json.loads(record.read_text())
    if chk["script"] != "g7_rows.py" or m.get("stage") != "g7_rows":
        die(f"--allow-rhel9 record {record} is not a g7_rows marker")
    if m.get("outcome") not in ("PASS", "PASS_PARTIAL_SCOPE"):
        die(f"--allow-rhel9 record outcome {m.get('outcome')!r} is not a bitwise PASS")
    if not RHEL9_OS.match(str(m.get("host", {}).get("os_release", ""))):
        die(f"--allow-rhel9 record host OS {m.get('host', {}).get('os_release')!r} is not RHEL 9")
    if "summary.json" not in m.get("outputs", {}):
        die("--allow-rhel9 record has no summary.json output")
    summ = json.loads((record.parent / "summary.json").read_text())
    rows = summ.get("rows") or []
    if summ.get("outcome") != m.get("outcome") or summ.get("custody_failures"):
        die("--allow-rhel9 summary outcome/custody failures disagree with a bitwise PASS")
    for r in rows:
        if r.get("comparison", {}).get("bitwise_equal_non_self") is not True:
            die(f"--allow-rhel9 row {r.get('key')} is not bitwise equal on non-self columns")
        if not RHEL9_OS.match(str(r.get("this_host", {}).get("os_release", ""))):
            die(f"--allow-rhel9 row {r.get('key')} was not recomputed on RHEL 9")
    methods = {r.get("method") for r in rows}
    if not {"ot", "bob"} <= methods:
        die(f"--allow-rhel9 record rows cover methods {sorted(methods)}; OT and BoB are both required")
    res = {"record": str(record), "sha256": got, "outcome": m.get("outcome"), "rows": len(rows),
           "methods": sorted(methods), "os_release": m["host"]["os_release"]}
    _RHEL9_OK[key] = res
    return res


def allowed_queues(wall_min: int, rhel9_record: Path | None, rhel9_sha: str | None = None) -> set:
    allowed = set(RHEL8_QUEUES)
    if rhel9_record is not None:
        validate_rhel9_record(rhel9_record, rhel9_sha)
        allowed |= set(RHEL9_QUEUES)
    if wall_min <= SHORT_LIMIT_MIN:
        allowed.add(SHORT_QUEUE)
    return allowed


def check_probe_age(rec: dict, wave: str):
    """N4: production uses only probes of the current wave id, at most PROBE_MAX_AGE_HOURS old."""
    if rec.get("wave") != wave:
        die(f"dispatching.json is for wave {rec.get('wave')!r}, not the current wave {wave!r}")
    if not rec.get("probes_submitted_earliest"):
        die(f"wave {wave} dispatching.json has no probes_submitted_earliest; re-probe (N4)")
    t_now = _dt.datetime.now().astimezone()
    for field in ("created", "probes_submitted_earliest"):
        age = t_now - parse_ts(rec[field])
        if age > _dt.timedelta(hours=PROBE_MAX_AGE_HOURS) or age < _dt.timedelta(minutes=-5):
            die(f"wave {wave} probes are stale: {field} {rec[field]} is {age} old (> {PROBE_MAX_AGE_HOURS} h); "
                f"run probe/probe-status for a new wave (N4)")


def resolve_queues(requested, *, wave: str, mem_mb: int, wall_min: int, go: bool,
                   rhel9_record: Path | None, rhel9_sha: str | None = None) -> list:
    allowed = allowed_queues(wall_min, rhel9_record, rhel9_sha)
    bad = [q for q in requested if q not in allowed]
    if bad:
        die(f"queues {bad} not permitted for -W {wall_min} (allowed {sorted(allowed)}; RHEL 9 only with "
            f"--allow-rhel9 <bitwise fidelity record>; short only for -W <= {SHORT_LIMIT_MIN})")
    status = enumerate_queues()
    usable = [q for q in requested if status.get(q) == "Open:Active"]
    disp = ROOT / "queue_probe" / wave / "dispatching.json"
    if disp.exists():
        rec = json.loads(disp.read_text())
        if rec["probe_mem_mb"] < mem_mb or rec["probe_wall_min"] < wall_min:
            die(f"wave {wave} probes (-M {rec['probe_mem_mb']} -W {rec['probe_wall_min']}) are smaller than "
                f"this stage's largest element (-M {mem_mb} -W {wall_min})")
        if go:
            check_probe_age(rec, wave)
        else:
            try:
                check_probe_age(rec, wave)
            except LaunchError as exc:
                print(f"[dry-run] warning (fatal at --go): {exc}")
        usable = [q for q in usable if q in rec["dispatching"]]
    elif go:
        die(f"no {disp}: run `probe` and `probe-status` for wave {wave} first")
    else:
        print(f"[dry-run] note: {disp} absent; queues not yet probe-filtered")
    if not usable:
        die("no usable queue after enumeration/probe filtering")
    return usable


def split_runs(n: int, queues: list, offset: int = 0) -> list:
    """Split element indices 1..n into contiguous near-equal chunks, one per queue (<= MAX_ARRAY)."""
    k = max(1, min(len(queues), n))
    k = max(k, math.ceil(n / MAX_ARRAY))
    size, rem = divmod(n, k)
    out, lo = [], 1
    for i in range(k):
        hi = lo + size + (1 if i < rem else 0) - 1
        out.append((queues[(offset + i) % len(queues)], lo, hi))
        lo = hi + 1
    return out


def index_spec(indices) -> str:
    """[3, 4, 5, 9] -> '3-5,9' (LSF array index list)."""
    idx = sorted(set(int(i) for i in indices))
    parts, start = [], None
    for i, v in enumerate(idx):
        if start is None:
            start = v
        if i + 1 == len(idx) or idx[i + 1] != v + 1:
            parts.append(f"{start}-{v}" if v != start else str(v))
            start = None
    return ",".join(parts)


# ------------------------------------------------------------------ paths / params
FEATQF = lambda: ROOT / "features_qf"                     # noqa: E731
VIEWS = lambda: ROOT / "similarity_inputs_qf"             # noqa: E731
MOSAIC = lambda: ROOT / "mosaic" / "attempt_001"          # noqa: E731
POPULATION = lambda: MOSAIC() / "retrieval_population.csv"  # noqa: E731
LOGS = lambda: ROOT / "logs_qf"                           # noqa: E731
PARAMS = lambda: write_base() / "params_qf"               # noqa: E731  (N5: scratch in dry runs)


def qf_task(b, f, arm, m) -> str:
    return f"{ROOT_NAME}/similarity_qf/budget{b}/fold{f}/{arm}_{m}"


def qf_query_ids(b, f) -> list:
    d = VIEWS() / "queries" / f"budget{b}" / f"fold{f}"
    return [Path(x).stem for x in sorted(glob.glob(str(d / "*.pkl")))]


def qf_library(b, f, arm) -> Path:
    return FEATQF() / f"budget{b}" / f"fold{f}" if arm == "both" else VIEWS() / UNFILTERED_VIEW


def qf_sim_line(b, f, arm, m, lo, hi) -> str:
    return (f"--encoder {ENCODER} --method {m} --features {VIEWS() / 'queries' / f'budget{b}' / f'fold{f}'} "
            f"--library-features {qf_library(b, f, arm)} --feature-dim {FEATURE_DIM} "
            f"--lo {lo} --hi {hi} --task {qf_task(b, f, arm, m)} --num-workers 1")


def runs_of(indices: list, batch: int) -> list:
    runs = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1 and len(runs[-1]) < batch:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def write_params(path: Path, lines: list) -> Path:
    data = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != data:
            die(f"{path} exists with different content; refusing to overwrite (use a new --params-tag)")
    else:
        path.write_text(data)
    return path


def params_lines(path: Path) -> list:
    return [x for x in path.read_text().splitlines() if x.strip()]


def line_range(line: str) -> tuple:
    f = line.split()
    return int(f[f.index("--lo") + 1]), int(f[f.index("--hi") + 1])


def line_task(line: str) -> str:
    f = line.split()
    return f[f.index("--task") + 1]


def provenance_dir(task: str) -> Path:
    """Where the frozen similarity_cli writes provenance_<LSB_JOBINDEX>.json (OUT/logs/<task with / -> _>)."""
    return OUT / "logs" / task.replace("/", "_")


def no_resource_override(a, stage: str):
    """Review 001 B3/#23: section 8 resources are fixed; only gap-fill memory_kill takes --mem-override."""
    if a.mem_override or a.wall_override:
        die(f"{stage}: --mem-override/--wall-override are refused (contract section 8 resources; a memory kill "
            f"is resubmitted only via --gap-fill --reason memory_kill --mem-override)")


# ------------------------------------------------------------------ N6: fidelity markers
def _rbfid_pins() -> dict:
    p = FIDELITY_SCRIPTS / "rbfid.py"
    if sha256_file(p) != FIDELITY_CODE_SHA["rbfid.py"]:
        die(f"{p} sha256 differs from the accepted rbfid.py")
    spec = importlib.util.spec_from_file_location("_route_b_rbfid_pins", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: (str(v[0]), v[1]) for k, v in mod.PIN.items()}


def check_fidelity_marker(path, allow_partial_scope: bool = False) -> dict:
    """N6: a fidelity output is usable by a production gate only if its marker is COMPLETE (sole marker), records
    no dev/test-only/pin-override use, was written by the accepted fidelity code, its SHA256SUMS/outputs match the
    bytes on disk, and every bound input that names or matches a section-2 pin equals that pin.

    allow_partial_scope admits only a g7_rows FAILED.json with outcome PASS_PARTIAL_SCOPE (used by --allow-rhel9)."""
    path = Path(path)
    if path.name not in ("COMPLETE.json", "FAILED.json") or not path.is_file():
        die(f"fidelity marker {path} is not an existing COMPLETE.json/FAILED.json")
    d = path.parent
    m = json.loads(path.read_text())
    other = d / ("FAILED.json" if path.name == "COMPLETE.json" else "COMPLETE.json")
    if other.exists():
        die(f"{d} holds both COMPLETE.json and FAILED.json")
    if path.name == "FAILED.json" and not (allow_partial_scope and m.get("stage") == "g7_rows"
                                           and m.get("outcome") == "PASS_PARTIAL_SCOPE"):
        die(f"{path} is a FAILED marker (outcome {m.get('outcome')!r})")
    for key in ("dev", "test_only", "pin_overrides"):
        if m.get(key):
            die(f"{path} records {key}={m.get(key)!r}; dev/test/pin-override outputs are refused by gates (N6)")
    if any(str(x).startswith(("--pin-override", "--dev")) for x in (m.get("argv") or [])):
        die(f"{path} argv records --pin-override/--dev (N6)")
    code = m.get("code_sha256") or {}
    names = {Path(p).name: (p, s) for p, s in code.items()}
    if len(code) != 2 or len(names) != 2 or "rbfid.py" not in names:
        die(f"{path} code_sha256 must list rbfid.py and exactly one fidelity script, got {sorted(names)}")
    script = next(n for n in names if n != "rbfid.py")
    for name, (p, s) in names.items():
        want = FIDELITY_SCRIPTS / name
        if name not in FIDELITY_CODE_SHA or Path(p).resolve() != want.resolve():
            die(f"{path}: code {p} is not the accepted {want}")
        if s != FIDELITY_CODE_SHA[name] or sha256_file(want) != s:
            die(f"{path}: code {name} sha256 {s} is not the accepted {FIDELITY_CODE_SHA[name]}")
    if script == "s0_census.py":
        # s0_census.py writes stage s0_<mode> (--mode choices uni2 / h_optimus_1) into ROOT/fidelity/s0_<mode>
        mode = m["mode"] if "mode" in m else (d.name[len("s0_"):] if d.name.startswith("s0_") else None)
        if mode not in ("uni2", "h_optimus_1") or m.get("stage") != f"s0_{mode}" or m.get("stage") != d.name:
            die(f"{path}: stage {m.get('stage')!r} does not match script {script} "
                f"(expected s0_<mode> equal to the marker directory {d.name!r})")
    elif m.get("stage") != Path(script).stem:
        die(f"{path}: stage {m.get('stage')!r} does not match script {script}")
    sums = d / "SHA256SUMS.txt"
    if not sums.is_file() or sha256_file(sums) != m.get("sha256sums_sha256"):
        die(f"{path}: SHA256SUMS.txt missing or its sha256 differs from the marker")
    listed = {}
    for line in sums.read_text().splitlines():
        sha, name = line.split("  ", 1)
        listed[name] = sha
    if listed != (m.get("outputs") or {}):
        die(f"{path}: SHA256SUMS.txt and marker outputs differ")
    for name, sha in listed.items():
        f = d / name
        if Path(name).name != name or not f.is_file() or sha256_file(f) != sha:
            die(f"{path}: output {name} missing or changed on disk")
    pins = _rbfid_pins()
    for name, b in (m.get("input_pins") or {}).items():
        bp, bs = str(b.get("path")), b.get("sha256")
        if name in pins and (bp, bs) != pins[name]:
            die(f"{path}: input {name} {(bp, bs)} differs from the section-2 pin {pins[name]} (pin override?)")
        for pn, (pp, psha) in pins.items():
            if (bp == pp or bs == psha) and (bp, bs) != (pp, psha):
                die(f"{path}: input {name} {(bp, bs)} partially matches pin {pn} {(pp, psha)}")
    return {"marker": str(path), "marker_sha256": sha256_file(path), "stage": m.get("stage"), "script": script,
            "outcome": m.get("outcome"), "host": m.get("host"), "checked": now()}


# ------------------------------------------------------------------ stages
def st_queues(a):
    status = enumerate_queues()
    allowed = allowed_queues(a.wall, a.allow_rhel9, a.allow_rhel9_sha)
    for q, s in sorted(status.items()):
        print(f"{q:24s} {s:14s} {'candidate' if q in allowed else '-'}")


def st_probe(a, sub: Submitter):
    allowed = allowed_queues(a.wall, a.allow_rhel9, a.allow_rhel9_sha)
    status = enumerate_queues()
    d = ROOT / "queue_probe" / a.wave
    for q in a.queues or sorted(allowed):
        if q not in allowed:
            die(f"queue {q} is not a permitted candidate")
        if status.get(q) != "Open:Active":
            print(f"[probe] {q}: {status.get(q, 'absent')} -> not probed")
            continue
        cmd = (f"{env_exports()}; hostname; grep '^PRETTY_NAME=' /etc/os-release; grep -m1 'model name' "
               f"/proc/cpuinfo; env | grep -E '{ENV_RECORD_PATTERN}' | sort; ls -l {PYTHON}")
        name = f"r21rb_probe_{a.wave}_{q}"
        argv = bsub_argv(name=name, queue=q, mem_mb=a.mem, wall_min=a.wall,
                         out=d / f"{q}.%J.out", err=d / f"{q}.%J.err", cmd=cmd, hold=False)
        print(shlex.join(argv))
        sub.lines.append(shlex.join(argv))
        if a.go:
            if any(r.get("action") == "probe" and r.get("job_name") == name for r in journal_records()):
                die(f"once-guard: probe {name} is already journaled; use a new --wave")
            d.mkdir(parents=True, exist_ok=True)
            m = re.search(r"Job <(\d+)>", run_cmd(argv, env=lsf_env()))
            if not m:
                die("probe bsub returned no job id")
            journal_append({"action": "probe", "stage": "probe", "wave": a.wave, "job_id": m.group(1),
                            "job_name": name, "queue": q, "mem_mb": a.mem,
                            "wall_min": a.wall, "held": False, "argv": argv, "cmd_sha256": sha256_text(cmd)})


def bjobs_elements(job_id: str) -> list:
    out = run_cmd(["bjobs", "-a", "-noheader", "-o", "jobid jobindex stat queue exec_host job_name delimiter='|'",
                   str(job_id)])
    rows = []
    for line in out.splitlines():
        f = line.strip().split("|")
        if len(f) >= 6 and f[0].strip().isdigit():
            rows.append({"job_id": f[0].strip(), "index": int(f[1]) if f[1].strip().isdigit() else 0,
                         "stat": f[2].strip(), "queue": f[3].strip(), "host": f[4].strip(),
                         "name": f[5].strip()})
    return rows


def st_probe_status(a):
    probes = [r for r in journal_records() if r.get("action") == "probe" and r.get("wave") == a.wave]
    if not probes:
        die(f"no journaled probes for wave {a.wave}")
    result = {}
    for r in probes:
        el = bjobs_elements(r["job_id"])
        stat = el[0]["stat"] if el else "UNKNOWN"
        result[r["queue"]] = {"job_id": r["job_id"], "stat": stat, "host": el[0]["host"] if el else "",
                              "submitted": r["ts"]}
    earliest = min(probes, key=lambda r: parse_ts(r["ts"]))["ts"]
    rec = {"wave": a.wave, "created": now(), "probes_submitted_earliest": earliest,
           "probe_mem_mb": min(r["mem_mb"] for r in probes),
           "probe_wall_min": min(r["wall_min"] for r in probes), "probes": result,
           "dispatching": sorted(q for q, v in result.items() if v["stat"] in ("RUN", "DONE"))}
    d = ROOT / "queue_probe" / a.wave
    d.mkdir(parents=True, exist_ok=True)
    (d / "dispatching.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=1))


def _resolve(a, mem, wall):
    return resolve_queues(a.queues, wave=a.wave, mem_mb=mem, wall_min=wall, go=a.go,
                          rhel9_record=a.allow_rhel9, rhel9_sha=a.allow_rhel9_sha)


def st_qf_features(a, sub: Submitter):
    no_resource_override(a, "qf-features")
    mem, wall = RES["features"]
    queues = _resolve(a, mem, wall)
    bl = " ".join(str(b) for b in a.budgets)
    ids = {}
    for i, f in enumerate(a.folds):
        args = (f"--mosaic-dir {MOSAIC()} --features-dir {FEATURES_DIR} --out-root {FEATQF()} "
                f"--head-fold {f} --budgets {bl} --min-tiles 8")
        if f == a.passthrough_fold:
            args += " --passthrough " + " ".join(F2_SLIDES)
        name = f"r21rb_features_f{f}"
        ids[f] = sub.submit(name=name, queue=queues[i % len(queues)], mem_mb=mem, wall_min=wall,
                            log_dir=LOGS() / "features",
                            cmd=worker_cmd("features", "prosper.evaluation.r2_1_filter_features_cli", args),
                            dependency=dep(a.after), label="features")
    return ids


def st_qf_views(a, sub: Submitter):
    no_resource_override(a, "qf-views")
    mem, wall = RES["views"]
    queues = _resolve(a, mem, wall)
    args = (f"--make-views --mosaic-dir {MOSAIC()} --features-dir {FEATURES_DIR} --out-root {FEATQF()} "
            f"--views-root {VIEWS()} --budgets {' '.join(map(str, a.budgets))} "
            f"--head-folds {' '.join(map(str, a.folds))}")
    return sub.submit(name="r21rb_views", queue=queues[0], mem_mb=mem, wall_min=wall, log_dir=LOGS() / "views",
                      cmd=worker_cmd("views", "prosper.evaluation.r2_1_filter_features_cli", args),
                      dependency=dep(a.after), label="views")


def sim_plan(a) -> list:
    """[(key, b, f, arm, m, params_path, n_elements)] -- params written (refuse-to-change) as in the reference."""
    plan = []
    for b in a.budgets:
        for arm in a.arms:
            for m in a.methods:
                for f in a.folds:
                    ids = qf_query_ids(b, f)
                    if not ids:
                        die(f"no query view for budget {b} fold {f}: run qf-features and qf-views first")
                    if not qf_library(b, f, arm).is_dir():
                        die(f"library {qf_library(b, f, arm)} missing")
                    runs = runs_of(list(range(1, len(ids) + 1)), BATCH[m])
                    p = write_params(PARAMS() / f"sim_budget{b}_fold{f}_{arm}_{m}{a.params_tag}.txt",
                                     [qf_sim_line(b, f, arm, m, r[0], r[-1]) for r in runs])
                    plan.append((f"budget{b}_{arm}_{m}", b, f, arm, m, p, len(runs)))
    return plan


def st_qf_sim(a, sub: Submitter) -> dict:
    no_resource_override(a, "qf-sim")
    jids, offset = {}, 0
    plan = sim_plan(a)
    for key, b, f, arm, m, p, n in plan:
        mem, wall = RES[("sim", arm, m)]
        queues = _resolve(a, mem, wall)
        name = f"r21rb_sim_b{b}_f{f}_{arm}_{m}"
        stage = f"sim_budget{b}_fold{f}_{arm}_{m}"
        print(f"[qf-sim] {stage}: {n} elements over {len(queues)} queue(s), -M {mem} -W {wall}")
        for q, lo, hi in split_runs(n, queues, offset):
            jids.setdefault(key, []).append(sub.submit(
                name=name, queue=q, mem_mb=mem, wall_min=wall, log_dir=LOGS() / stage,
                cmd=worker_cmd(stage, "prosper.embeddings.similarity_cli", params=p),
                array=f"{lo}-{hi}", dependency=dep(a.after), label=stage,
                extra={"params": str(p), "elements": [lo, hi]}))
        offset += 1
    return jids


def preserve_failed_provenance(p: Path, task: str, indices: list, reason: str) -> dict:
    """N8: copy the failed attempt's provenance files (and record absent ones) before a gap-fill resubmission."""
    src = provenance_dir(task)
    dest = (ROOT / "launcher" / "gapfill_preserved" / task.replace("/", "_")
            / f"{_dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}_{reason}")
    dest.mkdir(parents=True, exist_ok=False)
    copied, absent = {}, []
    for i in indices:
        f = src / f"provenance_{i}.json"
        if not f.is_file():
            absent.append(i)
            continue
        sha = sha256_file(f)
        shutil.copy2(f, dest / f.name)
        if sha256_file(dest / f.name) != sha:
            die(f"N8: copy of {f} does not verify")
        copied[str(i)] = sha
    man = {"created": now(), "params": str(p), "params_sha256": sha256_file(p), "task": task,
           "source_dir": str(src), "reason": reason, "indices": indices, "copied_sha256": copied,
           "absent": absent}
    with open(dest / "manifest.json", "x") as fh:
        fh.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return {"dir": str(dest), "manifest_sha256": sha256_file(dest / "manifest.json"), "copied": len(copied),
            "absent": absent}


def st_gap_fill(a, sub: Submitter) -> dict:
    """Resubmit only absent rows of journaled params files, once per (params, index) across ALL reasons (B3),
    when no element of any prior array for those params is still active. Resources (specification section 7):
    node_failure = original; walltime = original memory, doubled -W; memory_kill = --mem-override strictly
    greater than the original memory, original -W."""
    reason = a.reason
    if a.wall_override:
        die("gap-fill: --wall-override is refused (walltime doubles the original -W; contract section 7)")
    if reason == "memory_kill":
        if not a.mem_override:
            die("memory_kill gap-fill requires --mem-override (the next memory tier)")
    elif a.mem_override:
        die(f"gap-fill {reason}: --mem-override is refused (original memory is kept)")
    prior = [r for r in journal_records() if r.get("action") == "submit" and r.get("params")]
    by_params = {}
    for r in prior:
        by_params.setdefault(r["params"], []).append(r)
    done_fills = {(r["params"], int(i)): r["gap_fill"] for r in prior if r.get("gap_fill")
                  for i in r.get("gap_indices", [])}
    jids = {}
    for p, recs in sorted(by_params.items()):
        p = Path(p)
        r0 = next((r for r in recs if not r.get("gap_fill")), None)
        if r0 is None:
            die(f"{p.name}: no original (non-gap-fill) submission in the journal")
        if r0.get("params_sha256") and sha256_file(p) != r0["params_sha256"]:
            die(f"{p.name}: params file changed since submission")
        lines = params_lines(p)
        task = line_task(lines[0])
        feats = Path(lines[0].split()[lines[0].split().index("--features") + 1])
        ids = [Path(x).stem for x in sorted(glob.glob(str(feats / "*.pkl")))]
        have = {Path(x).stem[len("similarity_"):] for x in glob.glob(str(OUT / task / "similarity_*.pkl"))}
        missing = []
        for i, line in enumerate(lines, 1):
            lo, hi = line_range(line)
            if any(ids[q - 1] not in have for q in range(lo, hi + 1)):
                missing.append(i)
        if not missing:
            continue
        for r in recs:
            guard_own(r["job_id"], r["job_name"])
            active = [e for e in bjobs_elements(r["job_id"]) if e["stat"] in ACTIVE_STATES]
            if active:
                die(f"{p.name}: job {r['job_id']} still has {len(active)} PEND/RUN elements; no gap-fill")
        again = {i: done_fills[(str(p), i)] for i in missing if (str(p), i) in done_fills}
        if again:
            die(f"{p.name}: elements {sorted(again)} were already gap-filled ({sorted(set(again.values()))}); "
                f"each element is gap-filled at most once across all reasons; contract section 7 STOP")
        mem0, wall0 = int(r0["mem_mb"]), int(r0["wall_min"])
        if reason == "memory_kill":
            if a.mem_override <= mem0:
                die(f"memory_kill: --mem-override {a.mem_override} must be strictly greater than the original "
                    f"-M {mem0}")
            mem, wall = a.mem_override, wall0
        elif reason == "walltime":
            mem, wall = mem0, 2 * wall0
        else:
            mem, wall = mem0, wall0
        queues = _resolve(a, mem, wall)
        stage = r0.get("label") or (p.stem.replace(a.params_tag, "") if a.params_tag else p.stem)
        log_dir = Path(r0["log_dir"]) if r0.get("log_dir") else LOGS() / stage
        extra = {"params": str(p), "gap_fill": reason, "gap_indices": missing,
                 "gap_original_job": r0["job_id"], "gap_original_mem_mb": mem0, "gap_original_wall_min": wall0}
        if a.go:
            extra["gap_preserved"] = preserve_failed_provenance(p, task, missing, reason)
        jids[str(p)] = sub.submit(
            name=r0["job_name"], queue=queues[0], mem_mb=mem, wall_min=wall, log_dir=log_dir,
            cmd=worker_cmd(stage, "prosper.embeddings.similarity_cli", params=p),
            array=index_spec(missing), label=stage, extra=extra)
    return jids


def st_f2_rows(a, sub: Submitter):
    """F-2: all-true UNI2 queries (features_qf/passthrough) vs library_unfiltered_pop2604; 5 OT rows
    (first listed slide per fold) + one BoB element over all ten slides."""
    no_resource_override(a, "f2-rows")
    pt = FEATQF() / "passthrough"
    ids = sorted(f"{s}.pkl" for s in F2_SLIDES)
    if pt.is_dir():
        have = sorted(Path(x).name for x in glob.glob(str(pt / "*.pkl")))
        if have != ids:
            die(f"passthrough directory holds {have}, expected the ten F-2 slides")
    elif a.go:
        die(f"{pt} missing: run qf-features with --passthrough first")
    order = [Path(x).stem for x in ids]
    lib = VIEWS() / UNFILTERED_VIEW
    out = {}
    for m in a.methods:
        task = f"{ROOT_NAME}/fidelity/f2_rows/{m}"
        if m == "ot":
            rows = [order.index(s) + 1 for s in F2_OT_SLIDES]
            lines = [f"--encoder {ENCODER} --method ot --features {pt} --library-features {lib} --feature-dim "
                     f"{FEATURE_DIM} --lo {i} --hi {i} --task {task} --num-workers 1" for i in rows]
        else:
            lines = [f"--encoder {ENCODER} --method bob --features {pt} --library-features {lib} --feature-dim "
                     f"{FEATURE_DIM} --lo 1 --hi {len(order)} --task {task} --num-workers 1"]
        p = write_params(write_base() / "fidelity" / "params" / f"f2_rows_{m}.txt", lines)
        mem, wall = RES[f"f2_{m}"]
        queues = _resolve(a, mem, wall)
        out[m] = sub.submit(name=f"r21rb_f2_{m}", queue=queues[0], mem_mb=mem, wall_min=wall,
                            log_dir=ROOT / "fidelity" / "logs" / f"f2_{m}",
                            cmd=worker_cmd(f"f2_{m}", "prosper.embeddings.similarity_cli", params=p),
                            array=f"1-{len(lines)}", dependency=dep(a.after), label=f"f2_{m}",
                            extra={"params": str(p)})
    return out


def st_combine(a, sub: Submitter) -> dict:
    no_resource_override(a, "combine-qf")
    mem, wall = RES["combine"]
    queues = _resolve(a, mem, wall)
    after = json.loads(Path(a.after_map).read_text()) if a.after_map else {}
    jids, i = {}, 0
    for b in a.budgets:
        for arm in a.arms:
            for m in a.methods:
                key = f"budget{b}_{arm}_{m}"
                args = (f"combine-qf --arm {ARMS[arm]} --method {m} --budget {b} --population {POPULATION()} "
                        f"--rows-root {ROOT / 'similarity_qf'} --banks-root {FEATQF()} "
                        f"--out-dir {ROOT / 'similarity_matrix_qf'}")
                jids[key] = sub.submit(
                    name=f"r21rb_combine_b{b}_{arm}_{m}", queue=queues[i % len(queues)], mem_mb=mem,
                    wall_min=wall, log_dir=LOGS() / "combine",
                    cmd=worker_cmd("combine", "prosper.evaluation.r2_1_filter_eval_cli", args,
                                   pre=[sha_check([(POPULATION(), POPULATION_SHA)])]),
                    dependency=dep(after.get(key, [])), label="combine")
                i += 1
    return jids


def check_arm_bindings(arms: list, matrix_dir: Path) -> dict:
    """Addendum A2: assert every --arm tuple's label against its files before evaluate-qf.

    arms: [(name, method_tag, budget|'-', matrix, rows|'-')]."""
    checks = []
    tags = {v: k for k, v in EV.QF_METHOD_TAG.items()}
    for name, tag, budget, matrix, rows in arms:
        matrix = Path(matrix)
        rec = {"arm": name, "method": tag, "budget": budget, "matrix": str(matrix),
               "matrix_sha256": sha256_file(matrix)}
        if name == "all_tissue":
            if budget != "-" or rows != "-":
                die(f"all_tissue {tag}: budget/rows must be '-'")
            path, pin = ALL_TISSUE[tag]
            if matrix != path:
                die(f"(iii) all_tissue {tag} path {matrix} != pinned {path}")
            if rec["matrix_sha256"] != pin:
                die(f"(iii) all_tissue {tag} sha256 {rec['matrix_sha256']} != pin {pin}")
            rec["pinned_sha256"] = pin
        else:
            if name not in EV.QF_ARMS or tag not in tags or budget not in ("5", "100"):
                die(f"unknown arm tuple {(name, tag, budget)}")
            rows = Path(rows)
            want_m = EV.qf_matrix_name(budget, name, tag, masked=True)
            want_r = EV.qf_rows_name(budget, name, tag)
            if matrix.name != want_m or rows.name != want_r:
                die(f"(i) {name}/{tag}/{budget}: basenames {matrix.name}, {rows.name} != {want_m}, {want_r}")
            if matrix.parent != matrix_dir or rows.parent != matrix_dir:
                die(f"{name}/{tag}/{budget}: files are not in {matrix_dir}")
            crec_path = matrix_dir / (EV.qf_matrix_name(budget, name, tag, masked=False)[:-len(".pkl")]
                                      + "_combine_record.json")
            crec = json.loads(crec_path.read_text())
            declared = (crec.get("arm"), crec.get("method"), str(crec.get("budget")))
            if declared != (name, tags[tag], budget):
                die(f"(ii) {crec_path.name} declares {declared}, label is {(name, tags[tag], budget)}")
            # N1 (review 001 F5): the combine record must come from this package's rows/banks and method tag.
            for field, want in (("rows_root", str(ROOT / "similarity_qf")), ("banks_root", str(FEATQF())),
                                ("method_tag", tag)):
                if crec.get(field) != want:
                    die(f"(ii) {crec_path.name} {field} {crec.get(field)!r} != {want!r}")
            rec["rows_sha256"] = sha256_file(rows)
            for fname, field in ((matrix.name, "matrix_sha256"), (rows.name, "rows_sha256")):
                got = crec.get("outputs", {}).get(fname, {}).get("sha256")
                if got != rec[field]:
                    die(f"(ii) {crec_path.name} records {fname} sha256 {got} != file {rec[field]}")
            rec.update({"rows": str(rows), "combine_record": str(crec_path),
                        "combine_record_sha256": sha256_file(crec_path),
                        "combine_record_declares": {"arm": declared[0], "method": declared[1],
                                                    "budget": declared[2]},
                        "combine_rows_root": crec.get("rows_root"), "combine_banks_root": crec.get("banks_root"),
                        "combine_method_tag": crec.get("method_tag")})
        checks.append(rec)
    keys = [(c["arm"], c["method"], c["budget"]) for c in checks]
    if len(set(keys)) != len(keys):
        die("duplicate --arm label")
    return {"created": now(), "addendum": "A2", "assertions": ["(i) basenames", "(ii) combine record arm/method/"
            "budget + output sha256 + rows_root/banks_root/method_tag (N1)", "(iii) all_tissue pinned paths/sha256"],
            "arms": checks}


def standard_arms(matrix_dir: Path, budgets=BUDGETS) -> list:
    arms = [("all_tissue", tag, "-", str(ALL_TISSUE[tag][0]), "-") for tag in ("ot_cosine", "yottixel")]
    for b in budgets:
        for name in ("query_only", "both_filtered"):
            for tag in ("ot_cosine", "yottixel"):
                arms.append((name, tag, str(b), str(matrix_dir / EV.qf_matrix_name(str(b), name, tag, True)),
                             str(matrix_dir / EV.qf_rows_name(str(b), name, tag))))
    return arms


def st_evaluate(a, sub: Submitter) -> str:
    no_resource_override(a, "evaluate-qf")
    mem, wall = RES["evaluate"]
    queues = _resolve(a, mem, wall)
    mdir = ROOT / "similarity_matrix_qf"
    out_dir = ROOT / "evaluation_qf"
    arms = standard_arms(mdir)
    pending = ROOT / "launcher" / "arm_bindings_pending.json"
    pins = [(POPULATION(), POPULATION_SHA)] + [ALL_TISSUE[t][::1] for t in ("ot_cosine", "yottixel")]
    if mdir.is_dir():
        binding = check_arm_bindings(arms, mdir)
        for c in binding["arms"]:
            if c["arm"] != "all_tissue":
                pins += [(Path(c["matrix"]), c["matrix_sha256"]), (Path(c["rows"]), c["rows_sha256"])]
        if a.go:
            if pending.exists():
                die(f"{pending} exists; evaluate-qf is submitted once")
            once_guard("r21rb_evaluate", None, None, False)
            pending.parent.mkdir(parents=True, exist_ok=True)
            with open(pending, "x") as fh:
                fh.write(json.dumps(binding, indent=1, sort_keys=True) + "\n")
    elif a.go:
        die(f"{mdir} missing: combine-qf must finish before evaluate-qf (A2 assertions)")
    else:
        print(f"[dry-run] {mdir} absent: A2 assertions run at --go")
    if out_dir.exists() and any(out_dir.iterdir()):
        die(f"{out_dir} is not empty")
    args = " ".join(["evaluate-qf", "--population", str(POPULATION())] +
                    [x for arm in arms for x in ("--arm", *arm)] +
                    ["--banks-root", str(FEATQF()), "--out-dir", str(out_dir),
                     "--draw-count", "5000", "--seed", "123"])
    if "--allow-used-sentinel" in args or "--standard" in args:
        die("evaluate-qf must run without --standard/--allow-used-sentinel")
    post = [f"cp -n {pending} {out_dir / 'arm_bindings.json'}"]
    return sub.submit(name="r21rb_evaluate", queue=queues[0], mem_mb=mem, wall_min=wall,
                      log_dir=LOGS() / "evaluate",
                      cmd=worker_cmd("evaluate", "prosper.evaluation.r2_1_filter_eval_cli", args,
                                     pre=[sha_check(pins)], post=post),
                      dependency=dep(a.after), label="evaluate")


def st_single(a, sub: Submitter) -> str:
    """Generic single job for the analyst-prepared A-1/A-2/registry/S-0/near-tie/G-5/G-7/RER stages."""
    if not a.name or not a.module:
        die("single requires --name and --module")
    if re.search(r"(^|\s)--(pin-override|dev)(=|\s|$)", a.module_args or ""):
        die("single: --pin-override/--dev are never passed by the launcher (N6)")
    queues = _resolve(a, a.mem, a.wall)
    if a.module.endswith(".py"):
        cmd = " && ".join([env_exports(), host_preamble(a.name),
                           " ".join([PYTHON, a.module, a.module_args or ""]).strip()])
    else:
        cmd = worker_cmd(a.name, a.module, a.module_args or "")
    return sub.submit(name=f"r21rb_{a.name}", queue=queues[0], mem_mb=a.mem, wall_min=a.wall,
                      log_dir=ROOT / "launcher" / "logs" / a.name, cmd=cmd, dependency=dep(a.after), label=a.name)


def wave_jobs(wave: str) -> list:
    return [r for r in journal_records() if r.get("action") == "submit" and r.get("wave") == wave]


def unreleased_jobs(wave: str) -> list:
    released = {r["job_id"] for r in journal_records() if r.get("action") == "release"}
    return [r for r in wave_jobs(wave) if r["job_id"] not in released]


# ------------------------------------------------------------------ N2: bjobs -l readback
BJOBS_CONT = "\n" + " " * 21                 # LSF continuation indent of wrapped bjobs -l fields
BJOBS_SEP = re.compile(r"^-{10,}\s*$", re.M)
MEM_UNITS = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
LIMIT_HEADER = re.compile(r"[ \t]*[A-Z]+LIMIT(?:[ \t]+[A-Z]+LIMIT)*[ \t]*")
_LIMIT_ITEM = r"\d+(?:\.\d+)?(?: ?(?:[KMGT]B?|min)(?![A-Za-z]))?"
LIMIT_VALUES = re.compile(rf"[ \t]*{_LIMIT_ITEM}(?:[ \t]+{_LIMIT_ITEM})*[ \t]*")
LIMIT_ITEM = re.compile(r"(\d+(?:\.\d+)?)(?: ?([KMGT]B?|min)(?![A-Za-z]))?")
BJOBS_CMD_DISPLAY_MAX = 4095                 # release 005: LSF truncates the `Command <...>` field at this length
READBACK_RULE_005 = "release_005_script_fallback"


def _limit_value(block: str, name: str):
    """(number, unit) under the ``name`` column of a `bjobs -l` limit table: a header line of *LIMIT names
    (e.g. ``MEMLIMIT SWAPLIMIT``) followed by one value line, matched by header-token position. None (fail
    closed) unless exactly one such header carries ``name`` once and its value line has one item per name."""
    lines, hits = block.split("\n"), []
    for i, line in enumerate(lines[:-1]):
        if not LIMIT_HEADER.fullmatch(line) or name not in line.split():
            continue
        names = line.split()
        items = LIMIT_ITEM.findall(lines[i + 1]) if LIMIT_VALUES.fullmatch(lines[i + 1]) else []
        hits.append(items[names.index(name)] if len(items) == len(names) and names.count(name) == 1 else None)
    return hits[0] if len(hits) == 1 else None


def _argv_val(argv: list, flag: str):
    return argv[argv.index(flag) + 1] if flag in argv else None


def _file_ok(shown: str, declared: str, jid: str, idx) -> bool:
    if shown == declared:
        return True
    sub = declared.replace("%J", jid).replace("%I", str(idx if idx is not None else 0))
    return shown == sub


def _occurrences(hay: str, needle: str) -> int:
    """Number of (possibly overlapping) exact occurrences of ``needle`` in ``hay``."""
    n, i = 0, hay.find(needle)
    while needle and i >= 0:
        n, i = n + 1, hay.find(needle, i + 1)
    return n


def _command_via_script(r: dict, cmd: str, tag: str, fetch_script, entry: dict) -> list:
    """Release 005: the `Command <` field is a clean BJOBS_CMD_DISPLAY_MAX-char display prefix of ``cmd``;
    verify the full command against `bjobs -script <tag>` instead (exactly one verbatim occurrence)."""
    entry.update(command_display_truncated=True, command_verified_via=None, script_sha256=None)
    if fetch_script is None:
        return [f"{tag}: Command field display-truncated at {BJOBS_CMD_DISPLAY_MAX} chars and no "
                f"`bjobs -script` fallback is available (cmd sha256 {r.get('cmd_sha256')})"]
    try:
        script = fetch_script(tag)
    except Exception as e:                    # noqa: BLE001  fail closed on any retrieval failure
        return [f"{tag}: Command field display-truncated and `bjobs -script {tag}` could not be obtained "
                f"({type(e).__name__}: {str(e)[:200]})"]
    if not isinstance(script, str) or not script:
        return [f"{tag}: Command field display-truncated and `bjobs -script {tag}` returned no script"]
    entry["script_sha256"] = sha256_text(script)
    problems = []
    if sha256_text(cmd) != r.get("cmd_sha256"):
        problems.append(f"{tag}: journaled command sha256 differs from the declared packet (script fallback)")
    n = _occurrences(script, cmd)
    if n != 1:
        problems.append(f"{tag}: `bjobs -script {tag}` contains the journaled command {n} times (need exactly 1; "
                        f"cmd sha256 {r.get('cmd_sha256')})")
    if not problems:
        entry["command_verified_via"] = "bjobs -script"
    return problems


def check_readback(r: dict, text: str, fetch_script=None, command_checks: dict | None = None) -> list:
    """Compare every `bjobs -l` element block of one journaled submission with its declared packet.

    Release 005: when the `Command <` field is exactly the first BJOBS_CMD_DISPLAY_MAX chars of a longer
    journaled command followed by `>`, the command is verified via ``fetch_script(tag)`` (`bjobs -script`)
    instead; ``command_checks`` (if given) receives one entry per element tag recording the route."""
    jid, name, argv = r["job_id"], r["job_name"], r.get("argv") or []
    cmd = argv[-1] if argv else ""
    problems = []
    if any(x in ("-N", "-u", "-oo", "-eo") for x in argv[:-1]):
        problems.append(f"{jid}: journaled argv carries -N/-u/-oo/-eo")
    if sha256_text(cmd) != r.get("cmd_sha256"):
        problems.append(f"{jid}: journaled command sha256 differs from the declared packet")
    if "LSB_JOB_REPORT_MAIL=N" not in cmd:
        problems.append(f"{jid}: command does not export LSB_JOB_REPORT_MAIL=N")
    flat = text.replace(BJOBS_CONT, "")
    blocks = [b for b in BJOBS_SEP.split(flat) if "Job <" in b]
    want_idx = expand_spec(r["array"]) if r.get("array") else None
    seen = []
    for b in blocks:
        m = re.search(r"Job <(\d+)(?:\[(\d+)\])?>", b)
        if not m or m.group(1) != jid:
            problems.append(f"{jid}: readback lacks a block for Job <{jid}> (got {m.group(0) if m else None})")
            continue
        idx = int(m.group(2)) if m.group(2) else None
        seen.append(idx)
        tag = f"{jid}[{idx}]" if idx is not None else jid
        nm = re.search(r"Job Name <([^>]*)>", b)
        names_ok = {name, f"{name}[{idx}]", f"{name}[{r.get('array')}]"} if r.get("array") else {name}
        if not nm or nm.group(1) not in names_ok:
            problems.append(f"{tag}: readback lacks 'Job Name <{name}>' (got {nm.group(1) if nm else None!r})")
        st = re.search(r"Status <(\w+)>", b)
        if not st or st.group(1) != "PSUSP":
            problems.append(f"{tag}: not held (PSUSP); status {st.group(1) if st else None}")
        q = re.search(r"Queue <([^>]*)>", b)
        if not q or q.group(1) != r["queue"]:
            problems.append(f"{tag}: readback lacks 'Queue <{r['queue']}>' (got {q.group(1) if q else None!r})")
        k = b.find("Command <")
        shown = b[k + len("Command <"):k + len("Command <") + len(cmd)] if k >= 0 else ""
        entry = {"command_display_truncated": False, "command_verified_via": None, "script_sha256": None}
        if k < 0 or sha256_text(shown) != r.get("cmd_sha256") or b[k + len("Command <") + len(cmd):][:1] != ">":
            c0 = k + len("Command <")
            if (k >= 0 and len(cmd) > BJOBS_CMD_DISPLAY_MAX and shown[:BJOBS_CMD_DISPLAY_MAX] ==
                    cmd[:BJOBS_CMD_DISPLAY_MAX] and b[c0 + BJOBS_CMD_DISPLAY_MAX:][:1] == ">"):
                problems += _command_via_script(r, cmd, tag, fetch_script, entry)
            else:
                problems.append(f"{tag}: Command differs from the declared packet (cmd sha256 {r.get('cmd_sha256')})")
        else:
            entry["command_verified_via"] = "bjobs -l"
        if command_checks is not None:
            command_checks[tag] = entry
        for label, flag in (("Output File", "-o"), ("Error File", "-e")):
            fm = re.search(label + r"( \(overwrite\))? <([^>]*)>", b)
            if not fm:
                problems.append(f"{tag}: readback lacks '{label}'")
            elif fm.group(1):
                problems.append(f"{tag}: {label} is in overwrite mode")
            elif not _file_ok(fm.group(2), str(_argv_val(argv, flag)), jid, idx):
                problems.append(f"{tag}: {label} {fm.group(2)} != declared {_argv_val(argv, flag)}")
        rl = _limit_value(b, "RUNLIMIT")
        if not rl or rl[1] != "min" or abs(float(rl[0]) - float(r["wall_min"])) > 1e-9:
            problems.append(f"{tag}: RUNLIMIT {rl} != {r['wall_min']} min")
        ml = _limit_value(b, "MEMLIMIT")
        if (not ml or ml[1][:1] not in MEM_UNITS
                or abs(float(ml[0]) * MEM_UNITS[ml[1][:1]] - r["mem_mb"]) > 0.005 * r["mem_mb"]):
            problems.append(f"{tag}: MEMLIMIT {ml} != {r['mem_mb']} MB")
        ru = re.search(r"rusage\[mem=([\d.]+)", b)
        if not ru or abs(float(ru.group(1)) - r["mem_mb"]) > 1e-9:
            problems.append(f"{tag}: rusage[mem] {ru.group(1) if ru else None} != {r['mem_mb']}")
        if "span[hosts=1]" not in b:
            problems.append(f"{tag}: readback lacks span[hosts=1]")
        if "Notify when job" in b or "Mail <" in b:
            problems.append(f"{tag}: e-mail notification is set")
    if want_idx is None:
        if seen != [None]:
            problems.append(f"{jid}: expected exactly one non-array block, got {seen}")
    elif sorted(i for i in seen if i is not None) != sorted(want_idx) or None in seen:
        problems.append(f"{jid}: element indices {sorted(i for i in seen if i is not None)} != declared "
                        f"{r['array']}")
    return problems


def st_readback(a):
    jobs = unreleased_jobs(a.wave)
    for r in jobs:
        guard_own(r["job_id"], r["job_name"])
    if not jobs:
        die(f"wave {a.wave} has no unreleased journaled jobs")
    d = ROOT / "launcher" / "readback" / a.wave
    d.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")
    problems, files, fallback = [], {}, False
    for r in jobs:
        text = run_cmd(["bjobs", "-l", r["job_id"]])
        f = d / f"{r['job_id']}_{stamp}.txt"
        f.write_text(text)
        files[r["job_id"]] = {"path": str(f), "sha256": sha256_file(f)}
        scripts, checks = {}, {}

        def fetch_script(tag, scripts=scripts):
            """Release 005: `bjobs -script <jid>` / `<jid>[<idx>]`, saved beside the `bjobs -l` text."""
            s = run_cmd(["bjobs", "-script", tag], env=lsf_env())
            safe = tag.replace("[", "_").replace("]", "")
            sf = d / f"{safe}_{stamp}.script.txt"
            with open(sf, "xb") as fh:
                fh.write(s.encode())
            (d / f"{safe}.script.txt").write_bytes(s.encode())
            scripts[tag] = {"path": str(sf), "sha256": sha256_file(sf)}
            return s

        problems += check_readback(r, text, fetch_script=fetch_script, command_checks=checks)
        for tag, e in checks.items():
            if e["command_display_truncated"]:
                fallback = True
            if tag in scripts:
                e["script_path"] = scripts[tag]["path"]
                if scripts[tag]["sha256"] != e.get("script_sha256"):
                    problems.append(f"{tag}: saved `bjobs -script` file sha256 differs from the checked text")
        files[r["job_id"]]["command_checks"] = checks
    chk = {"wave": a.wave, "created": now(), "pass": not problems, "problems": problems,
           "job_ids": sorted(r["job_id"] for r in jobs), "readback_files": files,
           "journal_sha256": sha256_file(journal_path()), "launcher_sha256": launcher_sha()}
    if fallback:
        chk["readback_rule"] = READBACK_RULE_005
    text = json.dumps(chk, indent=1, sort_keys=True) + "\n"
    with open(d / f"readback_check_{stamp}.json", "x") as fh:
        fh.write(text)
    (d / "readback_check.json").write_text(text)
    if problems:
        die("; ".join(problems[:20]) + (f" ... ({len(problems)} problems)" if len(problems) > 20 else ""))
    print(f"[readback] wave {a.wave}: {len(jobs)} jobs held and matching their declared packets (PASS)")


def st_release(a):
    """B1: bresume only after a PASS readback whose job-ID set equals the wave's current unreleased set."""
    jobs = unreleased_jobs(a.wave)
    for r in jobs:
        guard_own(r["job_id"], r["job_name"])
    if not jobs:
        die(f"wave {a.wave} has no unreleased journaled jobs")
    chk_path = ROOT / "launcher" / "readback" / a.wave / "readback_check.json"
    if not chk_path.exists():
        die(f"wave {a.wave} has no readback; run readback first")
    chk = json.loads(chk_path.read_text())
    if chk.get("pass") is not True:
        die(f"wave {a.wave} readback did not PASS ({len(chk.get('problems') or [])} problems); no release")
    want = sorted(r["job_id"] for r in jobs)
    if chk.get("job_ids") != want:
        die(f"wave {a.wave} readback covered {chk.get('job_ids')} but the unreleased set is {want}; "
            f"run readback again (no job is released without its own readback)")
    chk_sha = sha256_file(chk_path)
    for r in jobs:
        if r["job_id"] not in chk["job_ids"]:
            die(f"job {r['job_id']} is not in the readback set")
        argv = ["bresume", r["job_id"]]
        print(shlex.join(argv))
        if a.go:
            run_cmd(argv, env=lsf_env())
            journal_append({"action": "release", "wave": a.wave, "job_id": r["job_id"],
                            "job_name": r["job_name"], "readback_check_sha256": chk_sha})


def st_migrate(a):
    """bswitch still-PEND elements of this wave's own jobs to --to; DONE/RUN elements are never moved."""
    jobs = wave_jobs(a.wave)
    if not jobs:
        die(f"no journaled jobs for wave {a.wave}")
    allowed = allowed_queues(max(r["wall_min"] for r in jobs), a.allow_rhel9, a.allow_rhel9_sha)
    if a.to not in allowed:
        die(f"target queue {a.to} not permitted")
    if enumerate_queues().get(a.to) != "Open:Active":
        die(f"target queue {a.to} is not Open:Active")
    moved = 0
    for r in wave_jobs(a.wave):
        guard_own(r["job_id"], r["job_name"])
        elements = bjobs_elements(r["job_id"])
        pend = [e for e in elements if e["stat"] == "PEND" and e["queue"] != a.to
                and not protected(e["job_id"], e["name"])]
        if a.max_elements:
            pend = pend[: max(0, a.max_elements - moved)]
        if not pend:
            continue
        target = f"{r['job_id']}[{index_spec([e['index'] for e in pend])}]" if r.get("array") else r["job_id"]
        argv = ["bswitch", a.to, target]
        print(shlex.join(argv))
        if a.go:
            run_cmd(argv, env=lsf_env())
            journal_append({"action": "migrate", "wave": a.wave, "job_id": r["job_id"],
                            "job_name": r["job_name"], "from": sorted({e["queue"] for e in pend}), "to": a.to,
                            "indices": [e["index"] for e in pend]})
        moved += len(pend)
    print(f"[migrate] {moved} PEND element(s) {'moved' if a.go else 'would move'} to {a.to}")


def st_status(a):
    rows = []
    for r in wave_jobs(a.wave):
        for e in bjobs_elements(r["job_id"]):
            rows.append(e)
    counts = {}
    for e in rows:
        counts[(e["queue"], e["stat"])] = counts.get((e["queue"], e["stat"]), 0) + 1
    for (q, s), n in sorted(counts.items()):
        print(f"{q:20s} {s:6s} {n}")
    d = ROOT / "launcher" / "status" / a.wave
    d.mkdir(parents=True, exist_ok=True)
    (d / f"status_{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(rows) + "\n")


def st_collect(a):
    """N7: at collection, write ROOT/launcher/provenance_sums/<stage>.txt (sha256sum format) over every
    provenance_<i>.json in the actual OUT/logs/<task with / -> _>/ location, for each params file of the wave."""
    params = sorted({r["params"] for r in wave_jobs(a.wave) if r.get("params")})
    if not params:
        die(f"wave {a.wave} has no journaled params-file submissions")
    d = ROOT / "launcher" / "provenance_sums"
    done = []
    for p in params:
        recs = [r for r in journal_records() if r.get("action") == "submit" and r.get("params") == p]
        for r in recs:
            guard_own(r["job_id"], r["job_name"])
            active = [e for e in bjobs_elements(r["job_id"]) if e["stat"] in ACTIVE_STATES]
            if active:
                die(f"{Path(p).name}: job {r['job_id']} still has {len(active)} active elements; collect after all "
                    f"elements finished")
        r0 = next((r for r in recs if not r.get("gap_fill")), recs[0])
        pp = Path(p)
        if r0.get("params_sha256") and sha256_file(pp) != r0["params_sha256"]:
            die(f"{pp.name}: params file changed since submission")
        lines = params_lines(pp)
        task = line_task(lines[0])
        src = provenance_dir(task)
        entries, missing = [], []
        for i in range(1, len(lines) + 1):
            f = src / f"provenance_{i}.json"
            if f.is_file():
                entries.append(f"{sha256_file(f)}  {f}\n")
            else:
                missing.append(i)
        label = (r0.get("label") or pp.stem) + (f".{a.collect_tag}" if a.collect_tag else "")
        dest = d / f"{label}.txt"
        text = "".join(entries)
        if dest.exists():
            if dest.read_text() != text:
                die(f"{dest} exists with different content (e.g. after a gap-fill); use --collect-tag")
            print(f"[collect] {dest} unchanged")
            continue
        write_new(dest, text)
        side = {"created": now(), "wave": a.wave, "params": p, "params_sha256": sha256_file(pp), "task": task,
                "provenance_dir": str(src), "expected": len(lines), "present": len(entries), "missing": missing,
                "jobs": [r["job_id"] for r in recs], "sums": str(dest), "sums_sha256": sha256_file(dest)}
        write_new(d / f"{label}.json", json.dumps(side, indent=1, sort_keys=True) + "\n")
        done.append((label, len(entries), len(missing)))
        print(f"[collect] {dest}: {len(entries)} provenance files, {len(missing)} missing")
    return done


def st_fidelity_check(a):
    if not a.marker:
        die("fidelity-check requires --marker <COMPLETE.json>")
    res = check_fidelity_marker(a.marker)
    d = ROOT / "launcher" / "fidelity_checks"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{_dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------ CLI
STAGES = ("queues", "probe", "probe-status", "qf-features", "qf-views", "qf-sim", "f2-rows", "combine-qf",
          "evaluate-qf", "single", "readback", "release", "migrate", "status", "collect", "fidelity-check")
SCRATCH_STAGES = ("queues", "probe", "qf-features", "qf-views", "qf-sim", "f2-rows", "combine-qf", "evaluate-qf",
                  "single", "release", "migrate")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--go", action="store_true", help="submit (held); default is a dry run")
    p.add_argument("--wave", default="", help="queue-probe wave label (required with --go)")
    p.add_argument("--queues", nargs="+", default=list(RHEL8_QUEUES))
    p.add_argument("--allow-rhel9", type=Path, default=None, metavar="G7_ROWS_MARKER",
                   help="lead only: admit rhel9_standard/rhel9_large_mem after a bitwise RHEL 9 OT+BoB g7_rows record")
    p.add_argument("--allow-rhel9-sha", default=None, help="sha256 of the --allow-rhel9 record (lead-supplied)")
    p.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS), choices=BUDGETS)
    p.add_argument("--folds", type=int, nargs="+", default=list(FOLDS), choices=FOLDS)
    p.add_argument("--arms", nargs="+", default=["both", "query"], choices=sorted(ARMS))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--params-tag", default="_bob25")
    p.add_argument("--passthrough-fold", type=int, default=0, choices=FOLDS)
    p.add_argument("--after", nargs="*", default=[])
    p.add_argument("--after-map", type=Path, default=None, help="combine-qf: JSON {budget{b}_{arm}_{m}: [ids]}")
    p.add_argument("--gap-fill", action="store_true", help="qf-sim: resubmit absent rows only (once per element)")
    p.add_argument("--reason", choices=("node_failure", "memory_kill", "walltime"), default="node_failure")
    p.add_argument("--mem-override", type=int, default=0, help="gap-fill memory_kill only: the next memory tier")
    p.add_argument("--wall-override", type=int, default=0, help="refused (kept so that its use is rejected)")
    p.add_argument("--mem", type=int, default=32768, help="probe/single -M (MB)")
    p.add_argument("--wall", type=int, default=900, help="probe/single -W (min)")
    p.add_argument("--name", default="")
    p.add_argument("--module", default="", help="single: python module (via worker) or script .py path")
    p.add_argument("--module-args", default="")
    p.add_argument("--to", default="", help="migrate: target queue")
    p.add_argument("--max-elements", type=int, default=0)
    p.add_argument("--scratch", type=Path, default=None, help="dry runs: write params/listings here (not under ROOT)")
    p.add_argument("--requires", type=Path, nargs="*", default=[], metavar="MARKER",
                   help="--go: fidelity COMPLETE.json markers that must pass the N6 check first")
    p.add_argument("--marker", type=Path, default=None, help="fidelity-check: marker to check")
    p.add_argument("--collect-tag", default="", help="collect: suffix for a second (e.g. post-gap-fill) collection")
    a = p.parse_args(argv)
    if a.go and not a.wave and a.stage not in ("queues",):
        p.error("--go requires --wave")
    return a


def main(argv=None) -> int:
    global _SCRATCH
    a = parse(argv)
    os.environ["LSB_JOB_REPORT_MAIL"] = "N"
    os.environ["PROSPER_OUT"] = str(OUT)
    _SCRATCH = None
    if not a.go and a.stage in SCRATCH_STAGES:
        _SCRATCH = init_scratch(a.scratch)
    sub = Submitter(a.go, a.wave, a.stage)
    if a.go:
        for mk in a.requires:
            sub.requires.append(check_fidelity_marker(mk))
    elif a.requires:
        print(f"[dry-run] --requires markers are checked at --go: {[str(x) for x in a.requires]}")
    fn = {"queues": lambda: st_queues(a), "probe": lambda: st_probe(a, sub),
          "probe-status": lambda: st_probe_status(a), "qf-features": lambda: st_qf_features(a, sub),
          "qf-views": lambda: st_qf_views(a, sub),
          "qf-sim": lambda: (st_gap_fill if a.gap_fill else st_qf_sim)(a, sub),
          "f2-rows": lambda: st_f2_rows(a, sub), "combine-qf": lambda: st_combine(a, sub),
          "evaluate-qf": lambda: st_evaluate(a, sub), "single": lambda: st_single(a, sub),
          "readback": lambda: st_readback(a), "release": lambda: st_release(a),
          "migrate": lambda: st_migrate(a), "status": lambda: st_status(a),
          "collect": lambda: st_collect(a), "fidelity-check": lambda: st_fidelity_check(a)}[a.stage]
    res = fn()
    if sub.lines:
        sub.finish()
    if isinstance(res, dict) and res:
        print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
