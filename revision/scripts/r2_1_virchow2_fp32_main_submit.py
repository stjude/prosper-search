#!/usr/bin/env python3
"""V2-main LSF submit wrapper (specification candidate 002: M0.1-M0.8, M1, M3, M4; addendum A7/A9).

A thin wrapper around the frozen `prosper_job.sh` + params-line pattern of `submit_similarity.py` (#23, which is
NOT executed: its FEATURE_DIM table rejects virchow2/virchow2_fp32 and its queue choice does not implement M0.4).

Subcommands (all LSF calls go through one `Lsf` object; `--dry-run` never calls LSF and writes nothing):
  prepare      write <stage>/jobs/<step>/{job.sh, params.txt, launch.sha256, step.json} (M0.2 hash guard,
               import-closure git diff, A9 host sidecar, env block of #44); never overwrites different bytes
  probe-wave   M0.4: `bqueues` -> <stage>/queue_probe/<wave>/bqueues.txt; one same-shape probe per candidate
               queue; probes.tsv; queues that dispatched within 15 min -> wave.json; afterwards ONLY this wave's
               own journaled probe ids still PEND are bkilled (F12)
  submit       HELD (`bsub -H`) submission split across the wave's dispatched queues, readback, `bresume`;
               every job id in <stage>/journal.jsonl; never duplicates an element already journaled. Readback (F17)
               waits up to READBACK_SETTLE_S for the PEND -> PSUSP propagation transient (every poll recorded);
               every submitted chunk is read back, passing ones released, then failures are named
  wave         probe-wave + submit
  ot-memory    canary rule [m8]: 32 GiB fan-out only if both OT canaries' LSF Max Memory <= 24,638 MB, else 40 GiB
  release      F11: readback + `bresume` of journaled held jobs without a `released` event; refused for any job
               whose readback failed
  retry        M0.5 single retry (64 GiB / 720 min) of an eligible element after a retry-shape probe wave; acts only
               when `bjobs` reports EXIT and the LSF output file exists (F2); otherwise stops writing nothing.
               Refused for RUN_STAGE_RESUBMIT_STEPS (use `resubmit`)
  resubmit     ADDENDUM_004: once-only analyst resubmission of an EXITed RUN_STAGE_RESUBMIT_STEPS step at its own
               production shape, reusing the prepared jobs/<step>/ files, after readback preconditions (every prior
               submission EXIT, no earlier lead_resubmit, outputs and stage markers absent, requires met, analyst reason
               file under <root>/lead/); journaled as `lead_resubmit`, then the same held -> readback -> release path
  migrate      move still-PEND elements of this package to a queue that dispatched this wave's probe (`bswitch`)
  monitor      M0.7 / A7 observer: state transitions -> <stage>/monitor.log; optional analyst-set A7 switching
Protected: job 326997606 and any job named *_DD_R1_* are never acted on (M0.8). No e-mail: every bsub has
-oo/-eo files and LSB_JOB_REPORT_MAIL=N is exported; -N/-u are refused.
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

REPO = '/PROSPER_WORKSPACE/prosper_revision'
OUT = os.path.join(REPO, 'prosper_revision_output')
ROOT = os.path.join(OUT, 'r2_1_virchow2_fp32_main_20261013')
PS = '/PROSPER_WORKSPACE/prosper-search'
PS_HEAD = 'fd50373ce2176cd77c04f2c932f963f6e5e7dcd1'
PY = '/HOME/miniconda3/bin/python3'
COMET = '/PROSPER_WORKSPACE/comet-feature-gen'
COMET_JOB_DEFAULT = '/PROSPER_WORKSPACE/comet-feature-gen'  # prosper_job.sh
TMPF = os.path.join(COMET, 'comet-virchow2', 'data', 'tmp_final')
FP32 = os.path.join(OUT, 'r2_1_virchow2_fp32_20261006', 'run_001', 'features', 'virchow2_fp32')
CONTRACT = os.path.join(ROOT, 'contract_candidate_002', 'CONTRACT.md')
CONTRACT_SHA = 'f0ef449ff2bd8708e07afa5a1e68a94d71e02ec3e057b2c6057e40fe7a98cca2'
WORKER = os.path.join(PS, 'scripts', 'lsf', 'prosper_job.sh')
SCRIPTS = {k: os.path.join(REPO, 'scripts', f'r2_1_virchow2_fp32_main_{k}.py')
           for k in ('fidelity', 'gates', 'rr5_adapter', 'compare', 'submit')}

FIDELITY_STAGE_A003 = 'fidelity_002'   # AMENDMENT_003 stage consumed by run-stage steps (m3_gates); fidelity-stage
#                                        steps incl. the m1_ranks_* ranks-precheck keep fidelity_001 (review F4)
RUN_STAGE = 'run_003'                  # ADDENDUM_003: every run-stage step resolves here; run_001 (m3_gates sampler
#                                        feature_dim spec error) and run_002 (m4_ranks_gate key-path defect) are
#                                        frozen FAILED. run_003 is created by `fidelity.py init-run` (hardlinks +
#                                        LINEAGE_RUN_003.json, chained to run_002/LINEAGE_RUN_002.json).
STAGES = ('fidelity_001', RUN_STAGE)
# AMENDMENT_004 / ADDENDUM_003: steps whose earlier-stage attempt is rejected (excluded) evidence and that are re-run
# in RUN_STAGE. The copied run_003/journal.jsonl still holds their run_001/run_002 submit records; for these steps
# only, a journaled submission counts toward "already journaled" only if its argv references RUN_STAGE (see
# journaled_elements). ADDENDUM_003 §2.4: m4_ranks_gate is added; m3_gates stays listed (literal ruling; its run_002
# record is therefore not counted in run_003 either -- see implementation_008 report, Issue C).
RUN_STAGE_RETRIED_STEPS = ('m3_gates', 'm4_ranks_gate')
# ADDENDUM_004 rulings 3-4: run-stage steps whose first RUN_STAGE submission EXITed and that may be resubmitted once,
# by the analyst, through `resubmit` (never `retry`/`submit`); the outputs each step's resubmission must not find present.
RUN_STAGE_RESUBMIT_STEPS = ('m4_adapter_run_a', 'm5_tie_hosts')
RESUBMIT_OUTPUTS = {'m4_adapter_run_a': ('rr5_additive_001', 'rr5_additive_001.full_spec.json',
                                         'rr5_additive_001_recompute_bindings'),
                    'm5_tie_hosts': ('tie_census_host_diversity.json',)}
RESUBMIT_ADDENDUM = 'ADDENDUM_004'
RR5_FP32_KEY = 'virchow2_fp32'    # mirror of the frozen rr5 adapter's FP32 (pins `packages` key); equality is tested
N_SLIDES = 2921
BOB_BATCH = 20
N_BOB = -(-N_SLIDES // BOB_BATCH)          # 147
OT_CANARY = (1349, 1682)
BOB_CANARY = (68,)
PROBE_INDICES = (911, 2431, 1682, 2585, 497, 1349, 509)
OT_MEM_LOW, OT_MEM_HIGH, OT_CANARY_LIMIT_MB = 32768, 40960, 24638
RETRY_MEM, RETRY_WALL = 65536, 720
RETRY_TERMS = ('TERM_MEMLIMIT', 'TERM_RUNLIMIT')
# F2 / ambiguity 27: 'closed' (a host admission state, also printed on requeue/pending lines) no longer qualifies;
# only lines of the final dispatch's event history are scanned (failure_reason), and only after bjobs says EXIT.
HOST_FAIL_RE = re.compile(r'(?i)\bhosts?\b[^.;\n]*\b(fail(?:ed|ure)?|unavailable|unreachable|down|lost)\b'
                          r'|\bTERM_HOST\b|\bnode failure\b')
BHIST_EVENT_RE = re.compile(r'^\s*\w{3} \w{3} +\d+ \d\d:\d\d:\d\d(?: \d{4})?: ')

RHEL8_QUEUES = ('standard', 'priority', 'heavy_io', 'large_mem', 'large_core_count', 'superdome')
RHEL9_QUEUES = ('rhel9_standard', 'rhel9_large_mem')
SHORT_QUEUE, SHORT_MAX_WALL = 'short', 30
GPU_QUEUE_RE = re.compile(r'(?i)gpu|dgx')
PROBE_WINDOW_S = 15 * 60
PROTECTED_JOB_IDS = frozenset({'326997606'})
PROTECTED_NAME_RE = re.compile(r'_DD_R1_')
MAX_ARRAY_SIZE = 1000          # LSF MAX_JOB_ARRAY_SIZE default; a analyst flag can raise it

CLOSURE = ('prosper/__init__.py', 'prosper/config.py', 'prosper/pipeline.py', 'prosper/embeddings',
           *(f'prosper/evaluation/{m}.py' for m in ('__init__', 'slide_headers', 'metrics', 'paths', 'resample',
                                                     'cli', 'slide_eval', 'metrics_cli')))
CLI_I1 = tuple(range(1, 23))        # I.1 #1-#22 (PS modules + prosper_job.sh); #23 is never executed
RR5_I1 = (24, 26)                   # RR-5 script + vote module (executed by the adapter/compare via import)

MODULES = {'aggregate': 'prosper.embeddings.aggregate_cli', 'similarity': 'prosper.embeddings.similarity_cli',
           'combine': 'prosper.embeddings.combine_similarity_cli', 'metrics': 'prosper.evaluation.metrics_cli'}
METHODS = ('median_pooling', 'sampler', 'ot_cosine', 'yottixel')


class SubmitError(Exception):
    """A STOP condition (specification §S) or a refused action."""


def now_iso():
    return _dt.datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def stage_dir(root, stage):
    if stage not in STAGES:
        raise SubmitError(f'unknown stage {stage} (run_001 is frozen FAILED; AMENDMENT_004 run stage is {RUN_STAGE})')
    return os.path.join(root, stage)


def assert_writable(path, root):
    rp = os.path.realpath(path)
    rr = os.path.realpath(root)
    if not (rp == rr or rp.startswith(rr + os.sep)):
        raise SubmitError(f'{path} is outside the package root {root}')
    rel = os.path.relpath(rp, rr)
    if rel.split(os.sep)[0].startswith('contract_'):
        raise SubmitError(f'refusing to write under {rel.split(os.sep)[0]} (N.1)')


def write_new(path, data, root):
    """Create `path` with `data`; identical existing bytes are accepted, different bytes are a STOP."""
    assert_writable(path, root)
    b = data.encode() if isinstance(data, str) else data
    if os.path.lexists(path):
        with open(path, 'rb') as fh:
            if fh.read() == b:
                return False
        raise SubmitError(f'refusing to overwrite {path} with different bytes')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'xb') as fh:
        fh.write(b)
    return True


def append_line(path, line, root):
    assert_writable(path, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as fh:
        fh.write(line.rstrip('\n') + '\n')


# ----------------------------------------------------------------------------------------------
# I.1 frozen code table (parsed from the hash-checked specification)
# ----------------------------------------------------------------------------------------------

def load_i1(contract=CONTRACT, contract_sha=CONTRACT_SHA):
    if sha256_file(contract) != contract_sha:
        raise SubmitError(f'contract sha256 mismatch: {contract}')
    rx = re.compile(r'^\| (\d+) \| `(PS|REPO)/([^`]+)`[^|]*\| `([0-9a-f]{64})` \|', re.M)
    base = {'PS': PS, 'REPO': REPO}
    out = {}
    for m in rx.finditer(open(contract).read()):
        n = int(m.group(1))
        if 1 <= n <= 26:
            out[n] = (os.path.join(base[m.group(2)], m.group(3)), m.group(4))
    if sorted(out) != list(range(1, 27)):
        raise SubmitError(f'I.1 parse found {sorted(out)}')
    return out


# ----------------------------------------------------------------------------------------------
# step definitions
# ----------------------------------------------------------------------------------------------

def params_line(encoder, method, features, lo, hi, task):
    """submit_similarity.py params-line form, plus the specification's --metric cosine --reg 0 (M1.3/M3.2)."""
    return (f'--encoder {encoder} --method {method} --features {features} --feature-dim 1280 '
            f'--metric cosine --reg 0 --lo {lo} --hi {hi} --task {task}')


def bob_rows(b):
    return BOB_BATCH * (b - 1) + 1, min(BOB_BATCH * b, N_SLIDES)


def _step(name, stage, kind, mem, wall, **kw):
    d = {'name': name, 'stage': stage, 'kind': kind, 'mem_mb': mem, 'wall_min': wall,
         'jobname': f'v2m_{name}', 'indices': [0], 'one_element_arrays': False, 'args': [], 'module': None,
         'params': None, 'i1': list(CLI_I1), 'own_scripts': [], 'pre': [], 'mem_from': None,
         'requires': [], 'jobfile': None}
    d.update(kw)
    return d


def build_steps(root=ROOT, tmpf=TMPF, fp32=FP32):
    fd, rd = os.path.join(root, 'fidelity_001'), os.path.join(root, RUN_STAGE)     # ADDENDUM_003: rd = run_003
    fd2 = os.path.join(root, FIDELITY_STAGE_A003)     # AMENDMENT_003: fidelity-dir consumers read fidelity_002
    S = {}
    S['m1_bank_check'] = _step('m1_bank_check', 'fidelity_001', 'script', 8192, 180, i1=[],
                               own_scripts=['fidelity'], args=[SCRIPTS['fidelity'], 'bank-check', '--stage-dir', fd,
                                                               '--tmpf', tmpf, '--fp32', fp32])
    for m in ('median_pooling', 'sampler'):
        S[f'm1_aggregate_{m}'] = _step(f'm1_aggregate_{m}', 'fidelity_001', 'cli', 8192, 60, module=MODULES['aggregate'],
                                       args=['--encoder', 'virchow2', '--method', m, '--metric', 'cityblock',
                                             '--fidelity-encoder', 'virchow2'])
        S[f'm1_rebuild_{m}'] = _step(f'm1_rebuild_{m}', 'fidelity_001', 'cli', 8192, 60, module=MODULES['aggregate'],
                                     args=['--encoder', 'virchow2', '--features', tmpf, '--method', m,
                                           '--matrix-model', 'virchow2_pubrebuild', '--min-slides', str(N_SLIDES)])
        S[f'm3_aggregate_{m}'] = _step(f'm3_aggregate_{m}', RUN_STAGE, 'cli', 8192, 60, module=MODULES['aggregate'],
                                       args=['--encoder', 'virchow2_fp32', '--features', fp32, '--method', m,
                                             '--metric', 'cityblock', '--matrix-model', 'virchow2_fp32',
                                             '--min-slides', str(N_SLIDES)])
    for meth, mem, wall in (('ot', 40960, 360), ('bob', 16384, 60)):
        task = f'similarity/virchow2_{meth}_probe'
        S[f'm1_{meth}_probe'] = _step(
            f'm1_{meth}_probe', 'fidelity_001', 'array', mem, wall, module=MODULES['similarity'],
            indices=list(PROBE_INDICES), one_element_arrays=True,
            params=[params_line('virchow2', meth, tmpf, i, i, task) for i in range(1, N_SLIDES + 1)])
        S[f'm1_combine_{meth}_probe'] = _step(
            f'm1_combine_{meth}_probe', 'fidelity_001', 'cli', 16384, 60, module=MODULES['combine'],
            args=['--encoder', 'virchow2', '--method', meth, '--similarity-dir',
                  os.path.join(fd, 'similarity', f'virchow2_{meth}_probe'), '--fidelity-reference'])
        S[f'm3_combine_{meth}'] = _step(
            f'm3_combine_{meth}', RUN_STAGE, 'cli', 16384, 60, module=MODULES['combine'],
            args=['--encoder', 'virchow2_fp32', '--method', meth, '--similarity-dir',
                  os.path.join(rd, 'similarity', f'virchow2_fp32_{meth}'), '--matrix-model', 'virchow2_fp32',
                  '--min-slides', str(N_SLIDES)])
    for m in METHODS:
        S[f'm1_ranks_{m}'] = _step(f'm1_ranks_{m}', 'fidelity_001', 'cli', 4096, 30, module=MODULES['metrics'],
                                   own_scripts=['fidelity'],
                                   pre=[[PY, SCRIPTS['fidelity'], 'ranks-precheck', '--stage-dir', fd]],
                                   args=['--model', 'virchow2', '--method', m, '--depth', '10', '--iter', '5000',
                                         '--task', 'c2_metric_suite'])
        S[f'm4_ranks_{m}'] = _step(f'm4_ranks_{m}', RUN_STAGE, 'cli', 4096, 30, module=MODULES['metrics'],
                                   args=['--model', 'virchow2_fp32', '--method', m, '--depth', '10', '--iter',
                                         '5000', '--task', 'c2_metric_suite'])
    ot_params = [params_line('virchow2_fp32', 'ot', fp32, i, i, 'similarity/virchow2_fp32_ot')
                 for i in range(1, N_SLIDES + 1)]
    bob_params = [params_line('virchow2_fp32', 'bob', fp32, *bob_rows(b), 'similarity/virchow2_fp32_bob')
                  for b in range(1, N_BOB + 1)]
    S['m3_ot_canary'] = _step('m3_ot_canary', RUN_STAGE, 'array', OT_MEM_HIGH, 360, module=MODULES['similarity'],
                              indices=list(OT_CANARY), one_element_arrays=True, params=ot_params)
    S['m3_ot_fanout'] = _step('m3_ot_fanout', RUN_STAGE, 'array', None, 360, module=MODULES['similarity'],
                              indices=[i for i in range(1, N_SLIDES + 1) if i not in OT_CANARY], params=ot_params,
                              mem_from='ot_fanout_memory_decision.json', requires=['ot_canary'])
    S['m3_bob_canary'] = _step('m3_bob_canary', RUN_STAGE, 'array', 16384, 480, module=MODULES['similarity'],
                               indices=list(BOB_CANARY), one_element_arrays=True, params=bob_params)
    S['m3_bob_fanout'] = _step('m3_bob_fanout', RUN_STAGE, 'array', 16384, 480, module=MODULES['similarity'],
                               indices=[b for b in range(1, N_BOB + 1) if b not in BOB_CANARY], params=bob_params,
                               requires=['bob_canary'])
    S['m3_gates'] = _step('m3_gates', RUN_STAGE, 'script', 16384, 60, i1=[], own_scripts=['gates', 'fidelity'],
                          args=[SCRIPTS['gates'], 'm3', '--run-dir', rd, '--fidelity-dir', fd2, '--amendment',
                                os.path.join(root, 'contract_amendment_004', 'AMENDMENT_004.md')])
    for st in STAGES:
        S[f'm0_input_bindings_{st[:3]}'] = _step(f'm0_input_bindings_{st[:3]}', st, 'script', 4096, 30, i1=[],
                                                 own_scripts=['gates', 'fidelity'],
                                                 args=[SCRIPTS['gates'], 'input-bindings', '--stage', st,
                                                       '--stage-dir', os.path.join(root, st)])
    S['m4_ranks_gate'] = _step('m4_ranks_gate', RUN_STAGE, 'script', 4096, 30, i1=[], own_scripts=['gates', 'fidelity'],
                               args=[SCRIPTS['gates'], 'm4-ranks', '--run-dir', rd])
    S['m5_tie_hosts'] = _step('m5_tie_hosts', RUN_STAGE, 'script', 4096, 30, i1=[], own_scripts=['gates', 'fidelity'],
                              args=[SCRIPTS['gates'], 'tie-hosts', '--run-dir', rd], requires=['rr5_run_a'])
    S['m4_evaluator'] = _step('m4_evaluator', RUN_STAGE, 'jobfile', 8192, 45, i1=[],
                              jobfile=os.path.join(rd, 'eval_job_v2main_001.lsf'), jobname='r21_v2main_fp32_eval')
    S['m4_adapter_run_a'] = _step('m4_adapter_run_a', RUN_STAGE, 'script', 4096, 60, i1=list(RR5_I1),
                                  own_scripts=['rr5_adapter'], args=[SCRIPTS['rr5_adapter'], 'run-a', '--run-dir', rd],
                                  requires=['eval_pins'])
    # ADDENDUM_005 ruling 1: Run B (option (ii), RUN_B_SIGNOFF.md); --feasibility keeps the adapter default
    S['m4_adapter_run_b'] = _step('m4_adapter_run_b', RUN_STAGE, 'script', 8192, 60, i1=list(RR5_I1),
                                  own_scripts=['rr5_adapter'],
                                  args=[SCRIPTS['rr5_adapter'], 'run-b', '--run-dir', rd, '--signoff',
                                        os.path.join(root, 'RUN_B_SIGNOFF.md')],
                                  requires=['rr5_run_a'])
    S['m5_compare'] = _step('m5_compare', RUN_STAGE, 'script', 8192, 60, i1=list(RR5_I1),
                            own_scripts=['compare', 'rr5_adapter'], args=[SCRIPTS['compare'], '--run-dir', rd],
                            requires=['rr5_run_a', 'rr5_run_b'])        # ADDENDUM_005 ruling 3
    for d in S.values():
        assert not PROTECTED_NAME_RE.search(d['jobname'])
    return S


# ----------------------------------------------------------------------------------------------
# job file generation (M0.1 env, M0.2 guard, m9 closure, A9 sidecar)
# ----------------------------------------------------------------------------------------------

SIDECAR_PY = r'''import json, os, platform, socket, sys, time
m = ''
try:
    for line in open('/proc/cpuinfo'):
        if line.startswith('model name'):
            m = line.split(':', 1)[1].strip(); break
except OSError as e:
    m = 'unavailable: %s' % e
try:
    rel = open('/etc/redhat-release').read().strip()
except OSError:
    rel = ''
d = {'hostname': socket.gethostname(), 'cpu_model_name': m, 'os_release': rel, 'platform': platform.platform(),
     'lsb_jobid': os.environ.get('LSB_JOBID'), 'lsb_jobindex': os.environ.get('LSB_JOBINDEX'),
     'lsb_queue': os.environ.get('LSB_QUEUE'), 'lsb_job_name': os.environ.get('LSB_JOBNAME'),
     'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
with open(sys.argv[1], 'x') as fh:
    json.dump(d, fh, indent=1, sort_keys=True)
'''

CLOSURE_PY = r'''import hashlib, importlib, json, sys
importlib.import_module(sys.argv[1])
d = {}
for k, mod in sorted(sys.modules.items()):
    if k == 'prosper' or k.startswith('prosper.'):
        f = getattr(mod, '__file__', None)
        if f:
            d[k] = {'file': f, 'sha256': hashlib.sha256(open(f, 'rb').read()).hexdigest()}
with open(sys.argv[2], 'x') as fh:
    json.dump(d, fh, indent=1, sort_keys=True)
'''


def guard_entries(step, i1, extra_files=()):
    ent = [(i1[n][0], i1[n][1]) for n in step['i1']]
    for k in step['own_scripts']:
        ent.append((SCRIPTS[k], sha256_file(SCRIPTS[k])))
    for p in extra_files:
        ent.append((p, sha256_file(p)))
    return ent


def render_job(step, sdir, guard_path, guard_sha, params_path=None):
    q = shlex.quote
    L = ['#!/bin/bash',
         f'# V2-main step {step["name"]} (generated by r2_1_virchow2_fp32_main_submit.py; contract candidate 002)',
         'set -euo pipefail',
         f'export PROSPER_OUT={q(sdir)}',
         f'export PROSPER_ROOT={q(PS)}',
         f'export PROSPER_PYTHON={q(PY)}',
         'export PYTHONHASHSEED=0', 'export PYTHONUNBUFFERED=1', 'export PYTHONDONTWRITEBYTECODE=1',
         'export TQDM_DISABLE=1', 'export CUDA_VISIBLE_DEVICES=""',
         *(f'export {v}=1' for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                     'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')),
         'export LSB_JOB_REPORT_MAIL=N',
         'unset COMET_ROOT']
    if step['kind'] == 'cli' or step['kind'] == 'array':
        L.append(f'export PROSPER_MODULE={q(step["module"])}')
    L.append(f'export PROSPER_PARAMS={q(params_path)}' if params_path else 'unset PROSPER_PARAMS')
    L += ['# M0.2 hash guard: exit 97 before any Python',
          f"printf '%s\\n' {q(guard_sha + '  ' + guard_path)} | sha256sum --check --strict --status - "
          "|| { echo 'V2M_GUARD: launch.sha256 changed' >&2; exit 97; }",
          f'sha256sum --check --strict --quiet {q(guard_path)} '
          "|| { echo 'V2M_GUARD: frozen/executed file sha256 mismatch' >&2; exit 97; }"]
    if step['kind'] in ('cli', 'array'):
        L += ['# m9 import closure: tracked closure files equal PS HEAD',
              f'git -C {q(PS)} diff --quiet {PS_HEAD} -- {" ".join(CLOSURE)} '
              "|| { echo 'V2M_GUARD: import-closure git diff non-empty' >&2; exit 97; }",
              '# prosper_job.sh exports its own COMET_ROOT default (a mount alias); it must resolve to COMET',
              f'[ "$(readlink -f {q(COMET_JOB_DEFAULT)})" = "$(readlink -f {q(COMET)})" ] '
              "|| { echo 'V2M_GUARD: COMET_ROOT default does not resolve to COMET' >&2; exit 97; }"]
    tag = f'{step["jobname"]}.${{LSB_JOBID:-none}}_${{LSB_JOBINDEX:-0}}'
    L += ['# A9 host sidecar (hostname + /proc/cpuinfo model name)',
          f'mkdir -p {q(os.path.join(sdir, "host_sidecar"))}',
          f'"$PROSPER_PYTHON" -I -S -c {q(SIDECAR_PY)} {q(os.path.join(sdir, "host_sidecar"))}/"{tag}.json"']
    if step['kind'] in ('cli', 'array'):
        L += ['# m9 {module: sha256} map of the executed CLI import closure (observed in a separate interpreter)',
              f'mkdir -p {q(os.path.join(sdir, "import_closure"))}',
              f'(cd {q(PS)} && "$PROSPER_PYTHON" -c {q(CLOSURE_PY)} {q(step["module"])} '
              f'{q(os.path.join(sdir, "import_closure"))}/"{tag}.json")']
    for pre in step['pre']:
        L.append(' '.join(q(x) for x in pre))
    if step['kind'] in ('cli', 'array'):
        L.append(f'exec /bin/bash {q(WORKER)}' + ''.join(' ' + q(x) for x in step['args']))
    elif step['kind'] == 'script':
        L += [f'cd {q(REPO)}', 'exec "$PROSPER_PYTHON" ' + ' '.join(q(x) for x in step['args'])]
    else:
        raise SubmitError(f'render_job: kind {step["kind"]}')
    return '\n'.join(L) + '\n'


def prepare(step, root=ROOT, i1=None, dry_run=False):
    sdir = stage_dir(root, step['stage'])
    jdir = os.path.join(sdir, 'jobs', step['name'])
    if step['kind'] == 'jobfile':
        if not os.path.exists(step['jobfile']):
            raise SubmitError(f'{step["name"]}: job file {step["jobfile"]} not yet written (adapter eval-spec)')
        rec = {'step': step['name'], 'kind': 'jobfile', 'job': step['jobfile'], 'job_sha256': sha256_file(step['jobfile'])}
        if not dry_run:
            write_new(os.path.join(jdir, 'step.json'), json.dumps(rec, indent=2, sort_keys=True) + '\n', root)
        return rec
    i1 = i1 if i1 is not None else load_i1()
    params_path = os.path.join(jdir, 'params.txt') if step['params'] else None
    params_txt = ('\n'.join(step['params']) + '\n') if step['params'] else None
    files = {}
    if params_txt:
        files['params.txt'] = params_txt
    guard = guard_entries(step, i1)
    for p, h in guard:
        if sha256_file(p) != h:
            raise SubmitError(f'M0.2: {p} sha256 differs from the contract I.1 value')
    if params_path:
        guard.append((params_path, hashlib.sha256(params_txt.encode()).hexdigest()))
    guard_txt = ''.join(f'{h}  {p}\n' for p, h in guard)
    guard_path = os.path.join(jdir, 'launch.sha256')
    guard_sha = hashlib.sha256(guard_txt.encode()).hexdigest()
    files['launch.sha256'] = guard_txt
    job = render_job(step, sdir, guard_path, guard_sha, params_path)
    files['job.sh'] = job
    rec = {'step': step['name'], 'stage': step['stage'], 'kind': step['kind'], 'jobname': step['jobname'],
           'module': step['module'], 'args': step['args'], 'n_elements': len(step['indices']),
           'mem_mb': step['mem_mb'], 'wall_min': step['wall_min'], 'mem_from': step['mem_from'],
           'one_element_arrays': step['one_element_arrays'],
           'files': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()},
           'guard': [{'path': p, 'sha256': h} for p, h in guard], 'contract_sha256': CONTRACT_SHA}
    files['step.json'] = json.dumps(rec, indent=2, sort_keys=True) + '\n'
    if not dry_run:
        for k, v in files.items():
            write_new(os.path.join(jdir, k), v, root)
    return rec


# ----------------------------------------------------------------------------------------------
# LSF access
# ----------------------------------------------------------------------------------------------

class Lsf:
    """All LSF interaction. `runner(argv, env) -> (rc, stdout, stderr)` is injectable for tests."""

    def __init__(self, runner=None, clock=None, sleep=None):
        self.runner = runner or self._subprocess
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep

    @staticmethod
    def _subprocess(argv, env):
        r = subprocess.run(argv, capture_output=True, text=True, env=env)
        return r.returncode, r.stdout, r.stderr

    def env(self):
        e = dict(os.environ)
        e['LSB_JOB_REPORT_MAIL'] = 'N'
        return e

    def call(self, argv, check=True):
        if argv[0] == 'bsub':
            check_bsub_argv(argv)
        rc, out, err = self.runner(list(argv), self.env())
        if check and rc != 0:
            raise SubmitError(f'{argv[0]} failed rc={rc}: {err.strip()[:400]}')
        return rc, out, err

    def bqueues(self):
        return self.call(['bqueues', '-w'])[1]

    def bsub(self, argv):
        out = self.call(argv)[1]
        m = re.search(r'Job <(\d+)> is submitted to (?:default )?queue <([^>]+)>', out)
        if not m:
            raise SubmitError(f'cannot parse bsub output: {out.strip()[:300]}')
        return m.group(1), m.group(2)

    def bjobs_rows(self, job_ids):
        """[{jobid, index, stat, queue, host, name, exit_code, start, exit_reason}] for our job ids."""
        ids = [str(j) for j in job_ids]
        for j in ids:
            guard_job(j)
        if not ids:
            return []
        fmt = "jobid jobindex stat queue exec_host job_name exit_code start_time exit_reason delimiter='|'"
        rc, out, err = self.call(['bjobs', '-a', '-noheader', '-o', fmt] + ids, check=False)
        rows = []
        for line in out.splitlines():
            f = line.split('|')
            if len(f) < 9 or not f[0].strip().isdigit():
                continue
            r = dict(zip(('jobid', 'index', 'stat', 'queue', 'host', 'name', 'exit_code', 'start', 'exit_reason'),
                         (x.strip() for x in f)))
            r['index'] = int(r['index']) if r['index'].isdigit() else 0
            if r['jobid'] in PROTECTED_JOB_IDS or PROTECTED_NAME_RE.search(r['name']):
                continue
            rows.append(r)
        return rows

    def bhist_l(self, job_id, index):
        guard_job(job_id)
        return self.call(['bhist', '-l', f'{job_id}[{index}]' if index else str(job_id)], check=False)[1]

    def bkill(self, job_id):
        guard_job(job_id)
        return self.call(['bkill', str(job_id)], check=False)


def check_bsub_argv(argv):
    if '-N' in argv or '-u' in argv:
        raise SubmitError('bsub -N/-u are forbidden (no e-mail)')
    if not ({'-o', '-oo'} & set(argv)) or not ({'-e', '-eo'} & set(argv)):
        if '<' not in argv:      # jobfile kind carries -oo/-eo inside the file (checked in submit_jobfile)
            raise SubmitError('every bsub must carry -o/-oo and -e/-eo files')
    if '-q' in argv and GPU_QUEUE_RE.search(argv[argv.index('-q') + 1]):
        raise SubmitError('GPU queues are never candidates (M0.4.5)')


def guard_job(job_id, name=''):
    if str(job_id) in PROTECTED_JOB_IDS or PROTECTED_NAME_RE.search(name or ''):
        raise SubmitError(f'protected job {job_id} {name} (M0.8)')


# ----------------------------------------------------------------------------------------------
# M0.4 queue probes
# ----------------------------------------------------------------------------------------------

def parse_bqueues(text):
    """{queue: status} from `bqueues -w` (header: QUEUE_NAME PRIO STATUS ...)."""
    out = {}
    for line in text.splitlines()[1:]:
        f = line.split()
        if len(f) >= 3:
            out[f[0]] = f[2]
    return out


def candidate_queues(wall_min, bq, allow_rhel9=False, include=None):
    """The CPU queues this element could run on: the six RHEL 8.10 production queues, `short` for <= 30-min
    steps, RHEL 9 queues only behind the analyst flag; only queues that `bqueues` lists as Open:Active."""
    cand = list(include) if include else list(RHEL8_QUEUES)
    if not include and wall_min <= SHORT_MAX_WALL:
        cand.append(SHORT_QUEUE)
    if allow_rhel9 and not include:
        cand += list(RHEL9_QUEUES)
    cand = [q for q in cand if not GPU_QUEUE_RE.search(q)]
    return [q for q in cand if bq.get(q, '').startswith('Open:Active')]


def rhel9_allowed(flag_path):
    """The analyst-set flag: a JSON file recording a bitwise RHEL 9 fidelity row (OT and BoB) [PROBE_RECORD]."""
    if not flag_path:
        return False
    d = json.load(open(flag_path))
    if not (d.get('lead_set') is True and d.get('ot_bitwise') is True and d.get('bob_bitwise') is True
            and d.get('os', '').startswith('RHEL 9')):
        raise SubmitError(f'RHEL 9 flag {flag_path} does not record a lead-set bitwise OT+BoB RHEL 9 fidelity row')
    return True


def probe_wave(lsf, root, stage, wave, mem_mb, wall_min, allow_rhel9=False, include=None, kind='production',
               poll_s=30, window_s=PROBE_WINDOW_S, dry_run=False):
    sdir = stage_dir(root, stage)
    wdir = os.path.join(sdir, 'queue_probe', wave)
    if dry_run:
        cand = candidate_queues(wall_min, {q: 'Open:Active' for q in (include or RHEL8_QUEUES + (SHORT_QUEUE,)
                                                                       + (RHEL9_QUEUES if allow_rhel9 else ()))},
                                allow_rhel9, include)
        for q in cand:
            print('DRY probe:', ' '.join(shlex.quote(x) for x in probe_argv(wdir, wave, q, mem_mb, wall_min)))
        return {'wave': wave, 'dispatched': cand, 'dry_run': True, 'mem_mb': mem_mb, 'wall_min': wall_min}
    if os.path.lexists(wdir):
        raise SubmitError(f'wave directory exists: {wdir}')
    assert_writable(wdir, root)
    os.makedirs(wdir)
    bq_text = lsf.bqueues()
    write_new(os.path.join(wdir, 'bqueues.txt'), bq_text, root)
    cand = candidate_queues(wall_min, parse_bqueues(bq_text), allow_rhel9, include)
    probes = {}
    for q in cand:
        jid, qq = lsf.bsub(probe_argv(wdir, wave, q, mem_mb, wall_min))
        probes[q] = {'job_id': jid, 'submit_t': lsf.clock(), 'dispatch_t': None, 'state': 'PEND', 'exit': ''}
        journal(root, stage, {'event': 'probe_submit', 'wave': wave, 'queue': q, 'job_id': jid,
                              'mem_mb': mem_mb, 'wall_min': wall_min, 'kind': kind})
    while True:
        rows = {r['jobid']: r for r in lsf.bjobs_rows([p['job_id'] for p in probes.values()])}
        t = lsf.clock()
        for q, p in probes.items():
            r = rows.get(p['job_id'])
            if r:
                p['state'] = r['stat']
                p['host'] = r['host']
                started = r['stat'] in ('RUN', 'DONE') or (r['stat'] == 'EXIT' and r['host'] not in ('', '-'))
                if started and p['dispatch_t'] is None:
                    p['dispatch_t'] = t
                p['exit'] = r['exit_code']
        undecided = [p for p in probes.values() if p['dispatch_t'] is None and t - p['submit_t'] < window_s]
        if not undecided:
            break
        lsf.sleep(poll_s)
    killed = kill_undispatched_probes(lsf, root, stage, wave, probes, window_s)
    lines = ['queue\tjob_id\tsubmit_time\tdispatch_time\tstate\texit\thost\tdispatched_within_15min']
    dispatched = []
    for q, p in probes.items():
        ok = p['dispatch_t'] is not None and p['dispatch_t'] - p['submit_t'] <= window_s
        if ok:
            dispatched.append(q)
        lines.append('\t'.join([q, p['job_id'], _ts(p['submit_t']), _ts(p['dispatch_t']), p['state'],
                                str(p['exit']), str(p.get('host', '')), str(ok)]))
    write_new(os.path.join(wdir, 'probes.tsv'), '\n'.join(lines) + '\n', root)
    rec = {'wave': wave, 'stage': stage, 'kind': kind, 'mem_mb': mem_mb, 'wall_min': wall_min,
           'candidates': cand, 'dispatched': dispatched, 'allow_rhel9': bool(allow_rhel9), 'time': now_iso(),
           'killed_undispatched_probes': killed,
           'note': ('F12 (LEAD_IMPL_DECISIONS_001): after the probe window, only this wave\'s own journaled probe ids '
                    'still PEND are bkilled; nothing else is touched')}
    write_new(os.path.join(wdir, 'wave.json'), json.dumps(rec, indent=2, sort_keys=True) + '\n', root)
    if not dispatched:
        raise SubmitError(f'wave {wave}: no candidate queue dispatched within 15 min')
    return rec


def kill_undispatched_probes(lsf, root, stage, wave, probes, window_s):
    """F12: bkill ONLY this wave's own journaled probe ids that never dispatched within the window and are still
    PEND on a fresh bjobs read. Every kill is journaled. Never a production element, 326997606 or *_DD_R1_*."""
    own = {r['job_id'] for r in read_journal(root, stage)
           if r.get('event') == 'probe_submit' and r.get('wave') == wave}
    cands = [p for p in probes.values() if p['dispatch_t'] is None and p['job_id'] in own]
    if not cands:
        return []
    fresh = {r['jobid']: r for r in lsf.bjobs_rows([p['job_id'] for p in cands])}
    killed = []
    for p in cands:
        r = fresh.get(p['job_id'])
        if r is None or r['stat'] != 'PEND' or not r['name'].startswith(f'v2m_probe_{wave}_'):
            continue
        rc, out, err = lsf.bkill(p['job_id'])
        p['state'] = 'PEND->bkill' if rc == 0 else f'PEND (bkill rc={rc})'
        journal(root, stage, {'event': 'probe_bkill', 'wave': wave, 'job_id': p['job_id'], 'rc': rc,
                              'window_s': window_s, 'reason': 'undispatched within the probe window (F12)'})
        killed.append({'job_id': p['job_id'], 'rc': rc})
    return killed


def _ts(t):
    return '' if t is None else _dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec='seconds')


def probe_argv(wdir, wave, q, mem_mb, wall_min):
    name = f'v2m_probe_{wave}_{q}'
    return ['bsub', '-q', q, '-J', name, '-n', '1', '-M', str(mem_mb), '-R', f'span[hosts=1] rusage[mem={mem_mb}]',
            '-W', str(wall_min), '-oo', os.path.join(wdir, f'{name}.%J.out'),
            '-eo', os.path.join(wdir, f'{name}.%J.err'), 'hostname; date; sleep 30']


def load_wave(root, stage, wave):
    """<stage>/queue_probe/<wave>/wave.json. ADDENDUM_003 (M0.4.2, a wave per stage): waves carried into this stage
    by lineage (recorded `stage` run_001/run_002) are refused, so only a wave probed for `stage` is used."""
    w = json.load(open(os.path.join(stage_dir(root, stage), 'queue_probe', wave, 'wave.json')))
    if w.get('stage') != stage:
        raise SubmitError(f'wave {wave} was probed for stage {w.get("stage")!r}, not {stage} (M0.4.2: a wave per '
                          'stage; probe a new one)')
    return w


# ----------------------------------------------------------------------------------------------
# journal / submission
# ----------------------------------------------------------------------------------------------

def journal(root, stage, rec):
    rec = dict(rec, time=rec.get('time') or now_iso())
    append_line(os.path.join(stage_dir(root, stage), 'journal.jsonl'), json.dumps(rec, sort_keys=True), root)


def read_journal(root, stage):
    p = os.path.join(stage_dir(root, stage), 'journal.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p) if x.strip()]


def _references_stage(rec, root, stage):
    """True if any argv element of a journaled submission lies under this stage directory (AMENDMENT_004)."""
    sd = os.path.realpath(stage_dir(root, stage)) + os.sep
    return any(isinstance(x, str) and os.path.realpath(x).startswith(sd) for x in rec.get('argv') or [])


def journaled_elements(root, stage, step_name):
    """{index: [submission records]} for a step (first submissions, retries and ADDENDUM_004 analyst resubmissions, so
    `submit` never re-submits an element a `lead_resubmit` record covers). AMENDMENT_004: for a
    RUN_STAGE_RETRIED_STEPS step in RUN_STAGE, submissions inherited from run_001 through the copied journal (argv
    referencing run_001, not RUN_STAGE) are not counted, so the gate-only retry can be submitted once in RUN_STAGE;
    every other step (fan-outs, aggregates, combines) still counts its inherited run_001 records (never re-run)."""
    d = {}
    retried = stage == RUN_STAGE and step_name in RUN_STAGE_RETRIED_STEPS
    for r in read_journal(root, stage):
        if r.get('event') in ('submit', 'retry_submit', 'lead_resubmit') and r.get('step') == step_name:
            if retried and not _references_stage(r, root, stage):
                continue
            for i in r['indices']:
                d.setdefault(i, []).append(r)
    return d


def index_spec(indices):
    """[1,2,3,5] -> '1-3,5'."""
    s = sorted(indices)
    parts, start, prev = [], s[0], s[0]
    for i in s[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(f'{start}' if start == prev else f'{start}-{prev}')
        if i is not None:
            start = prev = i
    return ','.join(parts)


def split_across(indices, queues, max_array=MAX_ARRAY_SIZE):
    """Contiguous balanced chunks, one or more per queue (each <= max_array elements)."""
    s = sorted(indices)
    n, k = len(s), len(queues)
    out, pos = [], 0
    for j, q in enumerate(queues):
        take = n // k + (1 if j < n % k else 0)
        chunk = s[pos:pos + take]
        pos += take
        for c in range(0, len(chunk), max_array):
            if chunk[c:c + max_array]:
                out.append((q, chunk[c:c + max_array]))
    assert sorted(i for _, ch in out for i in ch) == s
    return out


def element_bsub_argv(step, sdir, queue, indices, mem_mb, wall_min, held=True):
    jdir = os.path.join(sdir, 'jobs', step['name'])
    name = step['jobname'] + (f'[{index_spec(indices)}]' if step['kind'] == 'array' else '')
    argv = ['bsub'] + (['-H'] if held else []) + [
        '-q', queue, '-J', name, '-n', '1', '-M', str(mem_mb), '-R', f'span[hosts=1] rusage[mem={mem_mb}]',
        '-W', str(wall_min), '-cwd', REPO,
        '-oo', os.path.join(sdir, 'lsf', f'{step["jobname"]}.%J_%I.out'),
        '-eo', os.path.join(sdir, 'lsf', f'{step["jobname"]}.%J_%I.err'),
        '/bin/bash', os.path.join(jdir, 'job.sh')]
    return argv


def verify_prepared(step, root):
    sdir = stage_dir(root, step['stage'])
    jdir = os.path.join(sdir, 'jobs', step['name'])
    rec = json.load(open(os.path.join(jdir, 'step.json')))
    for k, h in rec.get('files', {}).items():
        if sha256_file(os.path.join(jdir, k)) != h:
            raise SubmitError(f'{step["name"]}: {k} changed since prepare')
    if step['kind'] != 'jobfile':
        # ADDENDUM_003 (implementation_008, defensive): a job directory carried by lineage from an earlier stage is
        # self-consistent but was prepared for that stage (its args name the earlier run dir); never submit it
        for k in ('stage', 'args', 'module', 'jobname'):
            if rec.get(k) != step[k]:
                raise SubmitError(f'{step["name"]}: prepared step.json {k} {rec.get(k)!r} != current {step[k]!r} '
                                  '(prepared for another stage or by another build; prepare anew in a new stage)')
    return rec


def resolve_mem(step, root):
    if step['mem_mb'] is not None:
        return step['mem_mb']
    p = os.path.join(stage_dir(root, step['stage']), step['mem_from'])
    if not os.path.exists(p):
        raise SubmitError(f'{step["name"]}: memory decision {p} missing (run ot-memory after the canaries)')
    return int(json.load(open(p))['fanout_mem_mb'])


def check_requires(step, root, ids=None):
    sdir = stage_dir(root, step['stage'])
    for req in step['requires']:
        if req == 'ot_canary':
            ot_memory_decision(root, write=False, ids=ids)
        elif req == 'bob_canary':
            canary_status(sdir, 'v2m_m3_bob_canary', BOB_CANARY,
                          [s for b in BOB_CANARY for s in range(bob_rows(b)[0], bob_rows(b)[1] + 1)],
                          'similarity/virchow2_fp32_bob', ids)
        elif req == 'eval_pins':
            # ADDENDUM_004 ruling 4: run-a needs the evaluator's COMPLETE marker and the M4 pins (FP32 package entry
            # + import closure) it reads
            if not os.path.isfile(os.path.join(sdir, 'eval_attempt_001', 'COMPLETE.json')):
                raise SubmitError(f'{step["name"]}: requires eval_pins: {sdir}/eval_attempt_001/COMPLETE.json absent')
            pp = os.path.join(sdir, 'pins_v2main_001.json')
            try:
                pins = json.load(open(pp))
            except (OSError, ValueError) as e:
                raise SubmitError(f'{step["name"]}: requires eval_pins: {pp} missing or not JSON ({e})')
            if not isinstance(pins, dict) or not isinstance(pins.get('packages'), dict) \
                    or RR5_FP32_KEY not in pins['packages'] or 'import_closure' not in pins:
                raise SubmitError(f'{step["name"]}: requires eval_pins: {pp} lacks packages[{RR5_FP32_KEY!r}] or '
                                  'import_closure')
        elif req == 'rr5_run_a':
            # ADDENDUM_004 ruling 4: tie-hosts / compare read run-a's RR-5 directory (COMPLETE + tie census)
            cp = os.path.join(sdir, 'rr5_additive_001', 'COMPLETE.json')
            try:
                c = json.load(open(cp))
            except (OSError, ValueError) as e:
                raise SubmitError(f'{step["name"]}: requires rr5_run_a: {cp} missing or not JSON ({e})')
            if not isinstance(c, dict) or c.get('status') != 'COMPLETE':
                raise SubmitError(f'{step["name"]}: requires rr5_run_a: {cp} status '
                                  f'{c.get("status") if isinstance(c, dict) else c!r} != COMPLETE')
            tc = os.path.join(sdir, 'rr5_additive_001', 'tie_census.csv')
            if not os.path.isfile(tc) or os.path.getsize(tc) == 0:
                raise SubmitError(f'{step["name"]}: requires rr5_run_a: {tc} absent or empty')
        elif req == 'rr5_run_b':
            # ADDENDUM_005 ruling 2: compare reads Run B's replacement products (faithfulness report passed, no FAILED)
            rb = os.path.join(sdir, 'rr5_replacement_001')
            fp = os.path.join(rb, 'faithfulness_report.json')
            try:
                r = json.load(open(fp))
            except (OSError, ValueError) as e:
                raise SubmitError(f'{step["name"]}: requires rr5_run_b: {fp} missing or not JSON ({e})')
            if not isinstance(r, dict) or r.get('pass') is not True:
                raise SubmitError(f'{step["name"]}: requires rr5_run_b: {fp} pass '
                                  f'{r.get("pass") if isinstance(r, dict) else r!r} is not true')
            if os.path.lexists(os.path.join(rb, 'FAILED.json')):
                raise SubmitError(f'{step["name"]}: requires rr5_run_b: {rb}/FAILED.json present')


def submit(lsf, root, step, wave, dry_run=False, max_array=MAX_ARRAY_SIZE, ids=None):
    sdir = stage_dir(root, step['stage'])
    if step['kind'] == 'jobfile':
        return submit_jobfile(lsf, root, step, wave, dry_run)
    mem = resolve_mem(step, root) if not dry_run or step['mem_mb'] else (step['mem_mb'] or OT_MEM_HIGH)
    if not dry_run:
        verify_prepared(step, root)
        check_requires(step, root, ids)
        w = load_wave(root, step['stage'], wave)
        if (w['mem_mb'], w['wall_min']) != (mem, step['wall_min']) or w.get('kind') != 'production':
            raise SubmitError(f'wave {wave} probed shape {w["mem_mb"]}/{w["wall_min"]} ({w.get("kind")}), '
                              f'step needs {mem}/{step["wall_min"]} (M0.4.2)')
        queues = w['dispatched']
    else:
        queues = list(RHEL8_QUEUES)
    done = journaled_elements(root, step['stage'], step['name'])
    todo = [i for i in step['indices'] if i not in done]
    if not todo:
        print(f'{step["name"]}: every element already journaled; nothing submitted')
        return []
    if step['one_element_arrays']:
        plan = [(queues[j % len(queues)], [i]) for j, i in enumerate(todo)]
    elif step['kind'] == 'array':
        plan = split_across(todo, queues, max_array)
    else:
        plan = [(queues[0], todo)]
    subs = []
    for q, idx in plan:
        argv = element_bsub_argv(step, sdir, q, idx, mem, step['wall_min'])
        if dry_run:
            print('DRY:', ' '.join(shlex.quote(x) for x in argv))
            continue
        os.makedirs(os.path.join(sdir, 'lsf'), exist_ok=True)
        jid, qq = lsf.bsub(argv)
        rec = {'event': 'submit', 'step': step['name'], 'wave': wave, 'job_id': jid, 'queue': qq,
               'indices': idx, 'mem_mb': mem, 'wall_min': step['wall_min'], 'argv': argv, 'held': True,
               'jobname': step['jobname']}
        journal(root, step['stage'], rec)
        subs.append(rec)
    release_submitted(lsf, root, step, subs)
    return subs


def release_submitted(lsf, root, step, subs):
    """F17: read back EVERY submitted chunk (each outcome journaled), release the passing ones, then stop naming the
    failed ids; a failed chunk stays held for the analyst. Shared by `submit` and `resubmit`."""
    failed = []
    for rec in subs:
        try:
            readback_and_release(lsf, root, step['stage'], rec)
        except SubmitError as e:
            failed.append((rec['job_id'], str(e)))
    if failed:
        ok = [r['job_id'] for r in subs if r['job_id'] not in {f[0] for f in failed}]
        raise SubmitError(f'{step["name"]}: readback/release failed for held job(s) {[f[0] for f in failed]} '
                          f'(left held for the lead); released {ok}; ' + ' | '.join(f[1] for f in failed))


READBACK_SETTLE_S = 60.0     # F17: bounded window for the PEND -> PSUSP propagation transient after `bsub -H`
READBACK_POLL_S = 2.5


def settle_held_rows(lsf, jid, window_s=READBACK_SETTLE_S, poll_s=READBACK_POLL_S):
    """F17: poll bjobs until no element of `jid` reports PEND. Only the all-PEND/PSUSP transient is waited out; any
    other state (RUN, DONE, EXIT, USUSP, SSUSP, ...) ends polling at once and is judged by the caller. Returns
    (final rows, settle record with every poll's raw rows)."""
    t0 = lsf.clock()
    max_polls = int(window_s / poll_s) + 2
    polls, timed_out = [], False
    while True:
        rows = lsf.bjobs_rows([jid])
        polls.append({'poll': len(polls) + 1, 't_s': round(lsf.clock() - t0, 3), 'rows': rows})
        stats = {r['stat'] for r in rows}
        if 'PEND' not in stats or stats - {'PEND', 'PSUSP'}:
            break
        if lsf.clock() - t0 >= window_s or len(polls) >= max_polls:
            timed_out = True
            break
        lsf.sleep(poll_s)
    return rows, {'n_polls': len(polls), 'settle_s': round(lsf.clock() - t0, 3), 'window_s': window_s,
                  'poll_s': poll_s, 'timed_out': timed_out, 'polls': polls}


def readback_and_release(lsf, root, stage, rec, attempt=None):
    jid = rec['job_id']
    rows, settle = settle_held_rows(lsf, jid)
    sdir = stage_dir(root, stage)
    rb_path = os.path.join(sdir, 'jobs', rec['step'], f'readback_{jid}.json' if attempt is None
                           else f'readback_{jid}_release{attempt:03d}.json')
    expect_idx = sorted(rec['indices']) if rec['indices'] != [0] else [0]
    problems = []
    if sorted(r['index'] for r in rows) != expect_idx:
        problems.append(f'indices {sorted(r["index"] for r in rows)[:10]}... != expected')
    for r in rows:
        if r['stat'] != 'PSUSP':
            problems.append(f'{jid}[{r["index"]}] stat {r["stat"]} != PSUSP')
        if r['queue'] != rec['queue']:
            problems.append(f'{jid}[{r["index"]}] queue {r["queue"]} != {rec["queue"]}')
        base = r['name'].split('[')[0]
        if base != rec['jobname']:
            problems.append(f'{jid}[{r["index"]}] name {r["name"]} != {rec["jobname"]}')
    if settle['timed_out']:
        problems.append(f'{jid}: PEND not settled to PSUSP within {settle["window_s"]} s '
                        f'({settle["n_polls"]} polls)')
    detail = lsf.call(['bjobs', '-l', jid], check=False)[1]
    write_new(rb_path, json.dumps({'rows': rows, 'problems': problems, 'bjobs_l': detail, 'settle': settle},
                                  indent=1, sort_keys=True) + '\n', root)
    sk = {'settle_polls': settle['n_polls'], 'settle_s': settle['settle_s']}
    if problems:
        journal(root, stage, dict({'event': 'readback_failed', 'step': rec['step'], 'job_id': jid,
                                   'problems': problems}, **sk))
        raise SubmitError(f'readback of held job {jid} failed (left held for the lead): {problems[:3]}')
    journal(root, stage, dict({'event': 'readback_ok', 'step': rec['step'], 'job_id': jid, 'n': len(rows)}, **sk))
    try:
        lsf.call(['bresume', jid])
    except SubmitError as e:
        journal(root, stage, {'event': 'release_failed', 'step': rec['step'], 'job_id': jid, 'error': str(e)[:400]})
        raise
    journal(root, stage, {'event': 'released', 'step': rec['step'], 'job_id': jid})


def release(lsf, root, stage, step_name=None, dry_run=False):
    """F11: re-run readback_and_release for journaled `held: true` submissions that have no `released` event.
    A record with a `readback_failed` event is refused (left held for the analyst; never released by the wrapper); the
    other records are still read back/released, then SubmitError names the refused ids."""
    jr = read_journal(root, stage)
    released = {r['job_id'] for r in jr if r.get('event') == 'released'}
    failed = {r['job_id'] for r in jr if r.get('event') == 'readback_failed'}
    todo = [r for r in jr if r.get('event') in ('submit', 'retry_submit') and r.get('held') is True
            and r['job_id'] not in released and (step_name is None or r.get('step') == step_name)]
    refused = [r['job_id'] for r in todo if r['job_id'] in failed]
    out = []
    for r in todo:
        if r['job_id'] in failed:
            continue
        guard_job(r['job_id'], r.get('jobname', ''))
        if dry_run:
            print(f'DRY: would read back and bresume held job {r["job_id"]} ({r["step"]})')
            continue
        n = sum(1 for x in jr if x.get('event') == 'release_attempt' and x.get('job_id') == r['job_id']) + 1
        journal(root, stage, {'event': 'release_attempt', 'step': r['step'], 'job_id': r['job_id'], 'attempt': n})
        readback_and_release(lsf, root, stage, r, attempt=n)
        out.append(r['job_id'])
    if refused and dry_run:
        print(f'DRY: release would be refused for held job(s) {refused} (readback failed)')
    elif refused:
        raise SubmitError(f'released {out}; release refused for held job(s) {refused}: their readback failed '
                          f'(left held for the lead)')
    return out


def submit_jobfile(lsf, root, step, wave, dry_run=False):
    """M4.2: the adapter-written job file (queue/-oo/-eo inside, #44 pattern). Its -q must be a queue that
    dispatched this wave's same-shape probe."""
    jf = step['jobfile']
    if not os.path.isfile(jf):
        raise SubmitError(f'{step["name"]}: job file {jf} not written yet (adapter m4 launch first)')
    txt = open(jf).read()
    q = re.search(r'^#BSUB -q (\S+)', txt, re.M)
    if not q or not re.search(r'^#BSUB -oo ', txt, re.M) or not re.search(r'^#BSUB -eo ', txt, re.M):
        raise SubmitError(f'{jf}: needs #BSUB -q/-oo/-eo')
    if re.search(r'^#BSUB -(N|u)\b', txt, re.M):
        raise SubmitError(f'{jf}: -N/-u forbidden')
    q = q.group(1)
    if dry_run:
        print(f'DRY: bsub -H < {jf}  (queue {q})')
        return []
    rec0 = json.load(open(os.path.join(stage_dir(root, step['stage']), 'jobs', step['name'], 'step.json')))
    if sha256_file(jf) != rec0['job_sha256']:
        raise SubmitError(f'{jf} changed since prepare')
    w = load_wave(root, step['stage'], wave)
    if q not in w['dispatched'] or (w['mem_mb'], w['wall_min']) != (step['mem_mb'], step['wall_min']):
        raise SubmitError(f'{jf}: queue {q} did not dispatch wave {wave} probe of shape {step["mem_mb"]}/'
                          f'{step["wall_min"]}')
    if journaled_elements(root, step['stage'], step['name']):
        print(f'{step["name"]}: already journaled; nothing submitted')
        return []
    rc, out, err = lsf.runner(['bash', '-c', f'bsub -H < {shlex.quote(jf)}'], lsf.env())
    m = re.search(r'Job <(\d+)> is submitted to (?:default )?queue <([^>]+)>', out)
    if rc != 0 or not m:
        raise SubmitError(f'bsub < {jf} failed: {err.strip()[:300]}')
    rec = {'event': 'submit', 'step': step['name'], 'wave': wave, 'job_id': m.group(1), 'queue': m.group(2),
           'indices': [0], 'mem_mb': step['mem_mb'], 'wall_min': step['wall_min'], 'argv': ['bsub', '-H', '<', jf],
           'held': True, 'jobname': re.search(r'^#BSUB -J (\S+)', txt, re.M).group(1)}
    journal(root, step['stage'], rec)
    readback_and_release(lsf, root, step['stage'], rec)
    return [rec]


# ----------------------------------------------------------------------------------------------
# canary memory policy [m8] and canary status
# ----------------------------------------------------------------------------------------------

def _fid():
    import importlib.util
    spec = importlib.util.spec_from_file_location('v2main_fidelity_helpers', SCRIPTS['fidelity'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sorted_ids(d):
    return sorted(x[:-4] for x in os.listdir(d) if x.endswith('.pkl'))


def canary_status(sdir, jobname, indices, rows_1based, task, ids=None):
    F = _fid()
    att = F.latest_attempts(sdir, jobname, 'out')
    ids = ids if ids is not None else sorted_ids(FP32)
    out = {}
    for i in indices:
        if i not in att:
            raise SubmitError(f'canary {jobname}[{i}] has no LSF out file yet')
        jid, p = att[i]
        d = F.parse_lsf_out(open(p, errors='replace').read())
        if d['rc'] != 0 or d['max_memory_mb'] is None:
            raise SubmitError(f'canary {jobname}[{i}] not successfully completed with a Max Memory line: {d}')
        out[i] = {'job_id': jid, 'out_file': p, **d}
    missing = [r for r in rows_1based
               if not os.path.exists(os.path.join(sdir, task, f'similarity_{ids[r - 1]}.pkl'))]
    if missing:
        raise SubmitError(f'canary rows missing for indices {missing[:10]}')
    return out


def ot_memory_decision(root, write=True, ids=None):
    sdir = stage_dir(root, RUN_STAGE)
    st = canary_status(sdir, 'v2m_m3_ot_canary', OT_CANARY, list(OT_CANARY), 'similarity/virchow2_fp32_ot', ids)
    peaks = {i: st[i]['max_memory_mb'] for i in OT_CANARY}
    mem = OT_MEM_LOW if all(v <= OT_CANARY_LIMIT_MB for v in peaks.values()) else OT_MEM_HIGH
    rec = {'rule': 'fan-out 32 GiB only if both canaries LSF Max Memory <= 24,638 MB, else 40 GiB [m8]',
           'canary_max_memory_mb': peaks, 'canary_jobs': {i: st[i]['job_id'] for i in OT_CANARY},
           'canary_hosts': {i: st[i]['host'] for i in OT_CANARY},
           'canary_run_time_s': {i: st[i]['run_time_s'] for i in OT_CANARY}, 'fanout_mem_mb': mem}
    if write:
        write_new(os.path.join(sdir, 'ot_fanout_memory_decision.json'), json.dumps(rec, indent=2, sort_keys=True)
                  + '\n', root)
    return rec


# ----------------------------------------------------------------------------------------------
# M0.5 retry
# ----------------------------------------------------------------------------------------------

def lsf_out_path(sdir, step, index, job_id):
    return os.path.join(sdir, 'lsf', f'{step["jobname"]}.{job_id}_{index}.out')


def element_terminal(lsf, sdir, step, index, job_id):
    """F2 precondition (bjobs -o stat): 'DONE', or 'EXIT' with the LSF output file present. Anything else
    (PEND/PSUSP/RUN/USUSP/SSUSP/UNKWN/ZOMBI, not found, or EXIT without the out file) raises SubmitError before
    anything is written."""
    rows = [r for r in lsf.bjobs_rows([job_id]) if r['index'] == index]
    if len(rows) != 1:
        raise SubmitError(f'{step["name"]}[{index}] job {job_id}: bjobs returned {len(rows)} rows; state unknown, '
                          f'nothing written')
    stat = rows[0]['stat']
    if stat == 'DONE':
        return 'DONE'
    if stat != 'EXIT':
        raise SubmitError(f'{step["name"]}[{index}] job {job_id} is {stat}, not EXIT/DONE; no retry, nothing written')
    if not os.path.exists(lsf_out_path(sdir, step, index, job_id)):
        raise SubmitError(f'{step["name"]}[{index}] job {job_id} is EXIT but its LSF output file is missing; '
                          f'nothing written')
    return 'EXIT'


def final_dispatch_lines(hist):
    """bhist -l lines of the last dispatch onward (continuation lines joined), so an earlier requeue's host line
    cannot qualify the final exit."""
    lines = re.sub(r'\n\s{20,}', '', hist).splitlines()
    starts = [i for i, ln in enumerate(lines) if BHIST_EVENT_RE.match(ln) and re.search(r':\s+Dispatched\b', ln)]
    return lines[starts[-1]:] if starts else lines


def failure_reason(lsf, sdir, step, index, job_id):
    """(eligible, reason, quoted_line). TERM_MEMLIMIT/TERM_RUNLIMIT from the LSF out file, or a host/node failure
    quoted from `bhist -l` (final dispatch only). Call only after element_terminal() returned EXIT."""
    F = _fid()
    p = lsf_out_path(sdir, step, index, job_id)
    d = F.parse_lsf_out(open(p, errors='replace').read()) if os.path.exists(p) else {'rc': None, 'term_reason': None}
    if d['rc'] == 0:
        return False, 'DONE', ''
    if d['term_reason'] in RETRY_TERMS:
        return True, d['term_reason'], ''
    hist = lsf.bhist_l(job_id, index)
    for line in final_dispatch_lines(hist):
        if HOST_FAIL_RE.search(line):
            return True, 'host_failure', line.strip()
    return False, d['term_reason'] or f'rc={d["rc"]}', ''


def retry(lsf, root, step, index, wave, dry_run=False):
    if step['name'] in RUN_STAGE_RESUBMIT_STEPS:
        raise SubmitError(f'{step["name"]} is an ADDENDUM_004 lead-resubmission step: M0.5 retry does not apply; '
                          f'use `resubmit --step {step["name"]} --wave W --reason-file PATH`; nothing written')
    sdir = stage_dir(root, step['stage'])
    if step['kind'] == 'jobfile':
        raise SubmitError('M0.5 retries apply to LSF-array/CLI steps of this wrapper only')
    subs = journaled_elements(root, step['stage'], step['name']).get(index, [])
    if not subs:
        raise SubmitError(f'{step["name"]}[{index}] was never submitted')
    first = subs[0]
    if any(s['event'] == 'retry_submit' for s in subs):
        last = subs[-1]
        if element_terminal(lsf, sdir, step, index, last['job_id']) == 'EXIT':
            elig, reason, line = failure_reason(lsf, sdir, step, index, last['job_id'])
            if reason != 'DONE':
                mark_failed(root, step, index, f'second failure ({reason}) after the single M0.5 retry')
        raise SubmitError(f'{step["name"]}[{index}] already retried once (M0.5)')
    if element_terminal(lsf, sdir, step, index, first['job_id']) == 'DONE':
        raise SubmitError(f'{step["name"]}[{index}] completed successfully; no retry')
    elig, reason, line = failure_reason(lsf, sdir, step, index, first['job_id'])
    if reason == 'DONE':
        raise SubmitError(f'{step["name"]}[{index}] completed successfully; no retry')
    if not elig:
        mark_failed(root, step, index, f'not eligible for the M0.5 retry: {reason}')
        raise SubmitError(f'{step["name"]}[{index}] failed with {reason}: element FAILED -> package FAILED')
    w = load_wave(root, step['stage'], wave)
    if w.get('kind') != 'retry' or (w['mem_mb'], w['wall_min']) != (RETRY_MEM, RETRY_WALL):
        raise SubmitError(f'wave {wave} is not a 64 GiB / 720 min retry-shape probe wave (M0.5)')
    n_prev = sum(1 for r in read_journal(root, step['stage']) if r.get('event') == 'retry_submit')
    q = w['dispatched'][n_prev % len(w['dispatched'])]
    argv = element_bsub_argv(dict(step, kind='array') if step['kind'] == 'array' else step, sdir, q, [index],
                             RETRY_MEM, RETRY_WALL)
    if dry_run:
        print('DRY retry:', ' '.join(shlex.quote(x) for x in argv))
        return None
    verify_prepared(step, root)
    jid, qq = lsf.bsub(argv)
    rec = {'event': 'retry_submit', 'step': step['name'], 'wave': wave, 'job_id': jid, 'queue': qq,
           'indices': [index], 'mem_mb': RETRY_MEM, 'wall_min': RETRY_WALL, 'argv': argv, 'held': True,
           'jobname': step['jobname'], 'first_job_id': first['job_id'], 'reason': reason, 'bhist_line': line}
    journal(root, step['stage'], rec)
    tsv = os.path.join(sdir, 'resubmissions.tsv')
    if not os.path.exists(tsv):
        append_line(tsv, 'element\tstep\tfirst_job_id\texit_reason\tbhist_line\tnew_job_id\tqueue\tmem_mb\twall_min',
                    root)
    append_line(tsv, '\t'.join(map(str, [index, step['name'], first['job_id'], reason, line.replace('\t', ' '),
                                         jid, qq, RETRY_MEM, RETRY_WALL])), root)
    readback_and_release(lsf, root, step['stage'], rec)
    return rec


# ----------------------------------------------------------------------------------------------
# ADDENDUM_004 analyst resubmission (rulings 3-4)
# ----------------------------------------------------------------------------------------------

def _load_reason(root, step_name, reason_file):
    """3f: (realpath, sha256, parsed dict, prior id set) of the analyst reason file; hash and parse the same bytes."""
    lead = os.path.realpath(os.path.join(root, 'lead'))
    rp = os.path.realpath(reason_file)
    if not rp.startswith(lead + os.sep):
        raise SubmitError(f'resubmit {step_name}: reason file {reason_file} is not under {lead}/; nothing written')
    if not os.path.isfile(rp):
        raise SubmitError(f'resubmit {step_name}: reason file {rp} is not a regular file; nothing written')
    b = open(rp, 'rb').read()
    try:
        d = json.loads(b)
    except ValueError as e:
        raise SubmitError(f'resubmit {step_name}: reason file {rp} is not JSON ({e}); nothing written')
    if not isinstance(d, dict):
        raise SubmitError(f'resubmit {step_name}: reason file {rp} is not a JSON object; nothing written')
    problems = []
    if d.get('step') != step_name:
        problems.append(f'step {d.get("step")!r} != {step_name!r}')
    if not isinstance(d.get('prior_job_ids'), list) or not d['prior_job_ids'] \
            or not all(isinstance(x, (int, str)) and not isinstance(x, bool) for x in d['prior_job_ids']):
        problems.append(f'prior_job_ids {d.get("prior_job_ids")!r} is not a non-empty list of job ids')
    if not isinstance(d.get('cause'), str) or not d['cause'].strip():
        problems.append('cause is not a non-empty string')
    if d.get('addendum') != RESUBMIT_ADDENDUM:
        problems.append(f'addendum {d.get("addendum")!r} != {RESUBMIT_ADDENDUM!r}')
    if problems:
        raise SubmitError(f'resubmit {step_name}: reason file {rp}: {"; ".join(problems)}; nothing written')
    return rp, hashlib.sha256(b).hexdigest(), d, {str(x) for x in d['prior_job_ids']}


def resubmit(lsf, root, step, wave, reason_file, dry_run=False):
    """ADDENDUM_004 rulings 3-4: once-only analyst resubmission of an EXITed run-stage step at its own production shape.
    Every precondition is read back before anything is written (else SubmitError, nothing written). --dry-run checks
    the file-only preconditions and prints the bsub line; it skips the bjobs-based 3a (and prior_exit_rc)."""
    name = step['name']
    if name not in RUN_STAGE_RESUBMIT_STEPS or step['kind'] != 'script' or step['indices'] != [0] \
            or step['stage'] != RUN_STAGE:
        raise SubmitError(f'resubmit applies only to the single-element {RUN_STAGE} script steps '
                          f'{RUN_STAGE_RESUBMIT_STEPS}, not {name}; nothing written')
    sdir = stage_dir(root, step['stage'])
    marks = [m for m in ('FAILED', 'COMPLETE.json', 'PARTIAL.json') if os.path.lexists(os.path.join(sdir, m))]
    if marks:                                                                                   # 3e (R7-1)
        raise SubmitError(f'resubmit {name}: stage {step["stage"]} carries {marks}; nothing written')
    jr = read_journal(root, step['stage'])
    if any(r.get('event') == 'lead_resubmit' and r.get('step') == name for r in jr):              # 3b
        raise SubmitError(f'resubmit {name}: a lead_resubmit record already exists (once only); nothing written')
    present = [p for p in RESUBMIT_OUTPUTS[name] if os.path.lexists(os.path.join(sdir, p))]
    if present:                                                                                 # 3c
        raise SubmitError(f'resubmit {name}: output path(s) {present} present in {sdir}; nothing written')
    reason_rp, reason_sha, _, reason_ids = _load_reason(root, name, reason_file)                  # 3f (fields)
    prior = journaled_elements(root, step['stage'], name).get(0, [])
    if not prior:
        raise SubmitError(f'resubmit {name}: no journaled submission in {step["stage"]}; nothing to resubmit '
                          '(use submit); nothing written')
    prior_ids = [str(r['job_id']) for r in prior]
    if reason_ids != set(prior_ids):                                                            # 3f (ids)
        raise SubmitError(f'resubmit {name}: reason-file prior_job_ids {sorted(reason_ids)} != journaled '
                          f'submissions {sorted(set(prior_ids))}; nothing written')
    try:
        verify_prepared(step, root)
        check_requires(step, root)                                                              # 3d
        w = load_wave(root, step['stage'], wave)
    except (OSError, ValueError) as e:
        raise SubmitError(f'resubmit {name}: prepared job files, requirement or wave {wave} unreadable ({e}); '
                          'nothing written')
    if (w['mem_mb'], w['wall_min']) != (step['mem_mb'], step['wall_min']) or w.get('kind') != 'production':
        raise SubmitError(f'wave {wave} probed shape {w["mem_mb"]}/{w["wall_min"]} ({w.get("kind")}), '
                          f'step needs {step["mem_mb"]}/{step["wall_min"]} (M0.4.2)')
    argv = element_bsub_argv(step, sdir, w['dispatched'][0], [0], step['mem_mb'], step['wall_min'])
    if dry_run:
        print('DRY resubmit (3a/prior_exit_rc not checked: no bjobs in dry-run):',
              ' '.join(shlex.quote(x) for x in argv))
        return None
    for r in prior:                                                                             # 3a
        st = element_terminal(lsf, sdir, step, 0, r['job_id'])
        if st != 'EXIT':
            raise SubmitError(f'resubmit {name}: journaled job {r["job_id"]} is {st}, not EXIT; nothing written')
    F = _fid()
    rcs = []
    for r in prior:
        p = lsf_out_path(sdir, step, 0, r['job_id'])
        try:
            rcs.append(F.parse_lsf_out(open(p, errors='replace').read())['rc'])
        except (OSError, KeyError):
            rcs.append(None)
    os.makedirs(os.path.join(sdir, 'lsf'), exist_ok=True)
    jid, qq = lsf.bsub(argv)
    rec = {'event': 'lead_resubmit', 'step': name, 'wave': wave, 'job_id': jid, 'queue': qq, 'indices': [0],
           'mem_mb': step['mem_mb'], 'wall_min': step['wall_min'], 'argv': argv, 'held': True,
           'jobname': step['jobname'], 'prior_job_ids': prior_ids, 'prior_exit_rc': rcs,
           'reason_file': reason_rp, 'reason_sha256': reason_sha}
    journal(root, step['stage'], rec)
    release_submitted(lsf, root, step, [rec])
    return rec


def mark_failed(root, step, index, why):
    rec = {'step': step['name'], 'index': index, 'why': why, 'time': now_iso()}
    append_line(os.path.join(stage_dir(root, step['stage']), 'failed_elements.jsonl'),
                json.dumps(rec, sort_keys=True), root)


# ----------------------------------------------------------------------------------------------
# migration / monitor (M0.7 + A7)
# ----------------------------------------------------------------------------------------------

def our_elements(root, stage):
    """{(job_id, index): submission record} for every journaled element of this package's stage."""
    d = {}
    for r in read_journal(root, stage):
        if r.get('event') in ('submit', 'retry_submit'):
            guard_job(r['job_id'], r.get('jobname', ''))
            for i in r['indices']:
                d[(r['job_id'], i)] = r
    return d


def switch_element(lsf, root, stage, job_id, index, target, wave, log_path, why):
    """bswitch ONE PEND element of this package to `target`, a queue that dispatched `wave`'s probe of the same
    shape. Re-reads the element state immediately before acting; never touches RUN/DONE/EXIT/PSUSP."""
    els = our_elements(root, stage)
    if (str(job_id), index) not in els:
        raise SubmitError(f'{job_id}[{index}] is not a journaled element of this package')
    rec = els[(str(job_id), index)]
    w = load_wave(root, stage, wave)
    if target not in w['dispatched'] or GPU_QUEUE_RE.search(target):
        raise SubmitError(f'{target} did not dispatch wave {wave}')
    if (w['mem_mb'], w['wall_min']) != (rec['mem_mb'], rec['wall_min']):
        raise SubmitError(f'wave {wave} shape {w["mem_mb"]}/{w["wall_min"]} != element shape '
                          f'{rec["mem_mb"]}/{rec["wall_min"]}')
    if target in RHEL9_QUEUES and not w.get('allow_rhel9'):
        raise SubmitError('RHEL 9 targets only behind the lead flag')
    cur = [r for r in lsf.bjobs_rows([job_id]) if r['index'] == index]
    if len(cur) != 1 or cur[0]['stat'] != 'PEND':
        return None
    guard_job(job_id, cur[0]['name'])
    if cur[0]['queue'] == target:
        return None
    spec = f'{job_id}[{index}]' if index else str(job_id)
    lsf.call(['bswitch', target, spec])
    line = '\t'.join([now_iso(), 'SWITCH', str(job_id), str(index), cur[0]['queue'], target, why])
    append_line(log_path, line, root)
    journal(root, stage, {'event': 'bswitch', 'job_id': str(job_id), 'index': index, 'from': cur[0]['queue'],
                          'to': target, 'wave': wave, 'why': why})
    return line


def migrate(lsf, root, stage, step_name, target, wave, indices=None):
    log_path = os.path.join(stage_dir(root, stage), 'monitor.log')
    els = {k: v for k, v in our_elements(root, stage).items() if v['step'] == step_name}
    rows = lsf.bjobs_rows(sorted({k[0] for k in els}))
    out = []
    for r in rows:
        k = (r['jobid'], r['index'])
        if k in els and r['stat'] == 'PEND' and (indices is None or r['index'] in indices):
            ln = switch_element(lsf, root, stage, r['jobid'], r['index'], target, wave, log_path, 'lead migrate')
            if ln:
                out.append(ln)
    return out


def monitor_once(lsf, root, stage, switch_after_min=None, state=None):
    """One poll: transitions -> monitor.log; summary of failures and PEND elements with queue/pending time.
    With `switch_after_min` (analyst-set; A7) PEND elements older than that are bswitch-ed to the dispatched queue of
    their own wave with the fewest PEND elements of this package."""
    sdir = stage_dir(root, stage)
    log_path = os.path.join(sdir, 'monitor.log')
    els = our_elements(root, stage)
    state = state if state is not None else {}
    rows = lsf.bjobs_rows(sorted({k[0] for k in els}))
    t = lsf.clock()
    fails, pend = [], []
    for r in rows:
        k = (r['jobid'], r['index'])
        if k not in els:
            continue
        key = f'{k[0]}[{k[1]}]'
        sig = (r['stat'], r['queue'], r['host'], r['exit_reason'])
        if state.get(key) != list(sig):
            append_line(log_path, '\t'.join([_ts(t), 'STATE', r['jobid'], str(r['index']), r['queue'], r['host'],
                                             r['stat'], r['exit_reason'] or '-']), root)
            state[key] = list(sig)
        if r['stat'] == 'EXIT':
            fails.append({'job_id': r['jobid'], 'index': r['index'], 'step': els[k]['step'],
                          'exit_reason': r['exit_reason']})
        if r['stat'] == 'PEND':
            sub_t = _dt.datetime.fromisoformat(els[k]['time']).timestamp()
            pend.append({'job_id': r['jobid'], 'index': r['index'], 'queue': r['queue'], 'step': els[k]['step'],
                         'wave': els[k]['wave'], 'pending_min': round((t - sub_t) / 60, 1)})
    switched = []
    if switch_after_min is not None:
        from collections import Counter
        load = Counter(p['queue'] for p in pend)
        for p in sorted(pend, key=lambda x: -x['pending_min']):
            if p['pending_min'] < switch_after_min:
                continue
            w = load_wave(root, stage, p['wave'])
            options = [q for q in w['dispatched'] if q != p['queue']]
            if not options:
                continue
            target = min(options, key=lambda q: (load[q], options.index(q)))
            ln = switch_element(lsf, root, stage, p['job_id'], p['index'], target, p['wave'], log_path,
                                f'A7 auto: pending {p["pending_min"]} min >= {switch_after_min}')
            if ln:
                load[p['queue']] -= 1
                load[target] += 1
                switched.append(ln)
    by_q = {}
    for p in pend:
        by_q.setdefault(p['queue'], []).append(p['pending_min'])
    summ = {'time': _ts(t), 'n_elements': len(els), 'n_rows': len(rows),
            'states': {s: sum(1 for r in rows if r['stat'] == s) for s in sorted({r['stat'] for r in rows})},
            'failures': fails, 'pending_by_queue': {q: {'n': len(v), 'max_pending_min': max(v)}
                                                    for q, v in sorted(by_q.items())},
            'switched': len(switched)}
    append_line(log_path, '\t'.join([_ts(t), 'SUMMARY', json.dumps(summ, sort_keys=True)]), root)
    return summ, state, pend


def monitor(lsf, root, stage, interval_s=300, once=False, switch_after_min=None):
    sp = os.path.join(stage_dir(root, stage), 'monitor_state.json')
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    while True:
        summ, state, pend = monitor_once(lsf, root, stage, switch_after_min, state)
        assert_writable(sp, root)
        with open(sp + '.tmp', 'w') as fh:
            json.dump(state, fh)
        os.replace(sp + '.tmp', sp)
        print(json.dumps(summ, sort_keys=True), flush=True)
        active = {k for k, v in summ['states'].items() if k in ('PEND', 'PSUSP', 'RUN', 'SSUSP', 'USUSP', 'WAIT')}
        if once or not active:
            return summ
        lsf.sleep(interval_s)


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--dry-run', action='store_true', help='print the bsub lines; call no LSF command; write nothing')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--step', nargs='+', required=True)
    for name in ('probe-wave', 'wave'):
        p = sub.add_parser(name)
        p.add_argument('--step', required=True)
        p.add_argument('--wave', required=True)
        p.add_argument('--rhel9-flag', default=None, help='lead-set JSON recording a bitwise RHEL 9 fidelity row')
        p.add_argument('--retry-shape', action='store_true', help='probe 64 GiB / 720 min (M0.5)')
        p.add_argument('--queues', nargs='*', default=None, help='restrict candidates (lead)')
        p.add_argument('--max-array-size', type=int, default=MAX_ARRAY_SIZE)
    p = sub.add_parser('submit')
    p.add_argument('--step', required=True)
    p.add_argument('--wave', required=True)
    p.add_argument('--max-array-size', type=int, default=MAX_ARRAY_SIZE)
    sub.add_parser('ot-memory')
    p = sub.add_parser('retry')
    p.add_argument('--step', required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--wave', required=True)
    p = sub.add_parser('resubmit', help='ADDENDUM_004: once-only lead resubmission of an EXITed run-stage step')
    p.add_argument('--step', required=True, choices=RUN_STAGE_RESUBMIT_STEPS)
    p.add_argument('--wave', required=True)
    p.add_argument('--reason-file', required=True)
    p = sub.add_parser('release', help='F11: readback + bresume journaled held jobs that were never released')
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--step', default=None)
    p = sub.add_parser('migrate')
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--step', required=True)
    p.add_argument('--to-queue', required=True)
    p.add_argument('--wave', required=True)
    p.add_argument('--indices', type=int, nargs='*')
    p = sub.add_parser('monitor')
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--interval', type=int, default=300)
    p.add_argument('--once', action='store_true')
    p.add_argument('--switch-after-min', type=float, default=None,
                   help='lead-set A7 threshold; omitted = observe only (no bswitch)')
    a = ap.parse_args(argv)
    steps = build_steps(a.root)
    lsf = Lsf()
    try:
        if a.cmd == 'prepare':
            i1 = load_i1()
            for s in a.step:
                rec = prepare(steps[s], a.root, i1, a.dry_run)
                print(json.dumps({'step': s, 'files': rec.get('files', rec.get('job_sha256'))}))
        elif a.cmd in ('probe-wave', 'wave'):
            st = steps[a.step]
            mem, wall = (RETRY_MEM, RETRY_WALL) if a.retry_shape else (
                (resolve_mem(st, a.root) if not a.dry_run or st['mem_mb'] else OT_MEM_HIGH), st['wall_min'])
            allow9 = rhel9_allowed(a.rhel9_flag)
            probe_wave(lsf, a.root, st['stage'], a.wave, mem, wall, allow9, a.queues,
                       'retry' if a.retry_shape else 'production', dry_run=a.dry_run)
            if a.cmd == 'wave' and not a.retry_shape:
                submit(lsf, a.root, st, a.wave, a.dry_run, a.max_array_size)
        elif a.cmd == 'submit':
            submit(lsf, a.root, steps[a.step], a.wave, a.dry_run, a.max_array_size)
        elif a.cmd == 'ot-memory':
            print(json.dumps(ot_memory_decision(a.root, write=not a.dry_run), indent=2, sort_keys=True))
        elif a.cmd == 'retry':
            retry(lsf, a.root, steps[a.step], a.index, a.wave, a.dry_run)
        elif a.cmd == 'resubmit':
            rec = resubmit(lsf, a.root, steps[a.step], a.wave, a.reason_file, a.dry_run)
            if rec:
                print(json.dumps({'lead_resubmit': rec['job_id'], 'step': a.step, 'queue': rec['queue']}))
        elif a.cmd == 'release':
            for j in release(lsf, a.root, a.stage, a.step, a.dry_run):
                print(json.dumps({'released': j}))
        elif a.cmd == 'migrate':
            if a.dry_run:
                print(f'DRY: would bswitch PEND elements of {a.step} to {a.to_queue}')
            else:
                for ln in migrate(lsf, a.root, a.stage, a.step, a.to_queue, a.wave, a.indices):
                    print(ln)
        elif a.cmd == 'monitor':
            if a.dry_run:
                print('DRY: monitor would poll bjobs for the journaled job ids of', a.stage)
            else:
                monitor(lsf, a.root, a.stage, a.interval, a.once, a.switch_after_min)
    except SubmitError as e:
        print(f'STOP: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
