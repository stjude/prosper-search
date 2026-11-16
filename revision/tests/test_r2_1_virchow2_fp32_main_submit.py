"""Synthetic tests for scripts/r2_1_virchow2_fp32_main_submit.py (no LSF: a fake runner simulates bqueues/bsub/
bjobs/bresume/bswitch/bhist). Run with an explicit --basetemp outside /tmp and PYTHONDONTWRITEBYTECODE=1."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('v2main_submit_under_test',
                                               os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_submit.py'))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

BQ = """QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS  PEND   RUN  SUSP
priority        150  Open:Active       -    -    -    -   10     0    10     0
standard        100  Open:Active       -    -    -    -   10     0    10     0
heavy_io        100  Open:Active       -    -    -    -   10     0    10     0
large_mem       100  Open:Active       -    -    -    -   10     0    10     0
large_core_count 100 Open:Active       -    -    -    -   10     0    10     0
superdome       100  Open:Inact        -    -    -    -   10     0    10     0
short           100  Open:Active       -    -    -    -   10     0    10     0
gpu             100  Open:Active       -    -    -    -   10     0    10     0
rhel9_standard  100  Open:Active       -    -    -    -   10     0    10     0
"""


class FakeLsf:
    """A minimal LSF: jobs {id: {name, queue, elements{idx: stat}}}; probes dispatch unless queue in `stuck`."""

    def __init__(self, stuck=(), protected=()):
        self.calls, self.envs = [], []
        self.jobs = {}
        self.next_id = 500
        self.stuck = set(stuck)
        self.t = 0.0
        for jid, name in protected:
            self.jobs[jid] = {'name': name, 'queue': 'gpu', 'el': {0: 'RUN'}, 'host': 'g1'}

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def __call__(self, argv, env):
        self.calls.append(list(argv))
        self.envs.append(env)
        cmd = argv[0]
        if cmd == 'bqueues':
            return 0, BQ, ''
        if cmd == 'bsub':
            q = argv[argv.index('-q') + 1]
            name = argv[argv.index('-J') + 1]
            held = '-H' in argv
            m = re.match(r'(.+)\[(.+)\]$', name)
            idx = [0]
            if m:
                name, spec = m.group(1), m.group(2)
                idx = []
                for part in spec.split(','):
                    a, _, b = part.partition('-')
                    idx += list(range(int(a), int(b or a) + 1))
            jid = str(self.next_id)
            self.next_id += 1
            probe = name.startswith('v2m_probe_')
            st = 'PSUSP' if held else ('RUN' if probe and q not in self.stuck else 'PEND')
            self.jobs[jid] = {'name': name, 'queue': q, 'el': {i: st for i in idx}, 'host': 'nodeA'}
            return 0, f'Job <{jid}> is submitted to queue <{q}>.\n', ''
        if cmd == 'bjobs':
            if '-l' in argv:
                return 0, f'Job <{argv[-1]}> details\n', ''
            out = []
            for jid in argv[5:]:
                j = self.jobs.get(jid)
                if not j:
                    continue
                for i, st in sorted(j['el'].items()):
                    nm = j['name'] + (f'[{i}]' if i else '')
                    host = j['host'] if st in ('RUN', 'DONE', 'EXIT') else '-'
                    out.append('|'.join([jid, str(i), st, j['queue'], host, nm, '-', '-', '-']))
            return 0, '\n'.join(out) + '\n', ''
        if cmd == 'bresume':
            for i in self.jobs[argv[1]]['el']:
                self.jobs[argv[1]]['el'][i] = 'PEND'
            return 0, '', ''
        if cmd == 'bswitch':
            q, spec = argv[1], argv[2]
            m = re.match(r'(\d+)(?:\[(\d+)\])?$', spec)
            j = self.jobs[m.group(1)]
            j.setdefault('el_queue', {})[int(m.group(2) or 0)] = q
            j['queue'] = q
            return 0, '', ''
        if cmd == 'bhist':
            return 0, getattr(self, 'bhist_text', ''), ''
        if cmd == 'bkill':
            for i in self.jobs[argv[1]]['el']:
                self.jobs[argv[1]]['el'][i] = 'EXIT'
            return 0, f'Job <{argv[1]}> is being terminated\n', ''
        raise AssertionError(argv)


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = str(tmp_path / 'pkg')
    os.makedirs(os.path.join(root, 'contract_candidate_002'))
    fake = FakeLsf(stuck={'large_mem'}, protected=[('326997606', 'gpu_job'), ('900', 'x_DD_R1_y')])
    lsf = S.Lsf(runner=fake, clock=fake.clock, sleep=fake.sleep)
    steps = S.build_steps(root, tmpf=str(tmp_path / 'tmpf'), fp32=str(tmp_path / 'fp32'))
    return {'root': root, 'fake': fake, 'lsf': lsf, 'steps': steps, 'tmp': tmp_path}


def fake_i1(tmp):
    d = {}
    for n in range(1, 27):
        p = os.path.join(str(tmp), f'frozen_{n}.py')
        open(p, 'w').write(f'# {n}\n')
        d[n] = (p, hashlib.sha256(open(p, 'rb').read()).hexdigest())
    return d


# ----------------------------------------------------------------------------------------------

def test_step_table_matches_contract(world):
    st = world['steps']
    assert st['m1_ot_probe']['indices'] == [911, 2431, 1682, 2585, 497, 1349, 509]
    assert st['m1_ot_probe']['one_element_arrays'] and st['m1_ot_probe']['mem_mb'] == 40960
    assert st['m1_bob_probe']['mem_mb'] == 16384 and st['m1_bob_probe']['wall_min'] == 60
    assert len(st['m1_ot_probe']['params']) == 2921
    assert st['m1_ot_probe']['params'][910].endswith('--lo 911 --hi 911 --task similarity/virchow2_ot_probe')
    assert st['m3_ot_canary']['indices'] == [1349, 1682] and st['m3_ot_canary']['mem_mb'] == 40960
    fan = st['m3_ot_fanout']
    assert len(fan['indices']) == 2919 and 1349 not in fan['indices'] and fan['mem_mb'] is None
    assert fan['wall_min'] == 360
    bob = st['m3_bob_fanout']
    assert len(bob['params']) == 147 and 68 not in bob['indices'] and len(bob['indices']) == 146
    assert bob['params'][67].split('--lo ')[1].startswith('1341 --hi 1360')
    assert bob['params'][146].split('--lo ')[1].startswith('2921 --hi 2921')
    assert st['m3_bob_canary']['indices'] == [68] and st['m3_bob_canary']['wall_min'] == 480
    assert st['m1_ranks_sampler']['wall_min'] == 30 and st['m1_ranks_sampler']['mem_mb'] == 4096
    assert st['m3_aggregate_sampler']['args'][-4:] == ['--matrix-model', 'virchow2_fp32', '--min-slides', '2921']
    assert '--fidelity-encoder' in st['m1_aggregate_sampler']['args']
    assert all('_DD_R1_' not in s['jobname'] for s in st.values())
    for k, sub in (('m4_ranks_gate', 'm4-ranks'), ('m5_tie_hosts', 'tie-hosts')):
        assert st[k]['args'][1] == sub and st[k]['wall_min'] <= 30 and st[k]['mem_mb'] == 4096
    assert 'package' not in {a for s in st.values() for a in s['args']}      # package COMPLETE is analyst-run
    assert st['m0_input_bindings_fid']['stage'] == 'fidelity_001' and st['m0_input_bindings_run']['args'][1:4] == [
        'input-bindings', '--stage', S.RUN_STAGE]
    line = st['m3_ot_canary']['params'][0]
    assert line.startswith('--encoder virchow2_fp32 --method ot --features ') and '--feature-dim 1280' in line


def test_load_i1_from_real_contract():
    i1 = S.load_i1()
    assert sorted(i1) == list(range(1, 27))
    assert i1[22][0].endswith('scripts/lsf/prosper_job.sh') and i1[24][0].endswith('r2_1_main_grid_plurality.py')
    assert i1[1][1] == 'e9b72617a564c598753ce06a68db0fae3ad2288667b3610c237f82aa301b0df7'


def test_prepare_job_files_and_never_overwrite(world):
    i1 = fake_i1(world['tmp'])
    st = world['steps']['m3_ot_canary']
    rec = S.prepare(st, world['root'], i1)
    jdir = os.path.join(world['root'], S.RUN_STAGE, 'jobs', 'm3_ot_canary')
    job = open(os.path.join(jdir, 'job.sh')).read()
    for needle in ('export PYTHONHASHSEED=0', 'export CUDA_VISIBLE_DEVICES=""', 'export OMP_NUM_THREADS=1',
                   'export OPENBLAS_NUM_THREADS=1', 'export MKL_NUM_THREADS=1', 'export LSB_JOB_REPORT_MAIL=N',
                   'unset COMET_ROOT', 'exit 97', f'diff --quiet {S.PS_HEAD} -- prosper/__init__.py',
                   'prosper/evaluation/metrics_cli.py', 'export PROSPER_MODULE=prosper.embeddings.similarity_cli',
                   'host_sidecar', 'import_closure', 'prosper_job.sh', 'export PROSPER_PARAMS='):
        assert needle in job, needle
    assert 'COMET_ROOT=' not in job.replace('unset COMET_ROOT', '')
    assert subprocess.run(['bash', '-n', os.path.join(jdir, 'job.sh')]).returncode == 0
    guard = open(os.path.join(jdir, 'launch.sha256')).read().splitlines()
    assert len(guard) == 22 + 1 and guard[-1].endswith('params.txt')
    assert S.prepare(st, world['root'], i1) == rec            # identical bytes: accepted
    open(os.path.join(jdir, 'params.txt'), 'a').write('x\n')
    with pytest.raises(S.SubmitError, match='overwrite'):
        S.prepare(st, world['root'], i1)
    with pytest.raises(S.SubmitError, match='changed since prepare'):
        S.verify_prepared(st, world['root'])


def test_prepare_refuses_hash_mismatch(world):
    i1 = fake_i1(world['tmp'])
    open(i1[3][0], 'a').write('edit\n')
    with pytest.raises(S.SubmitError, match='M0.2'):
        S.prepare(world['steps']['m1_aggregate_sampler'], world['root'], i1)


def test_refuses_contract_dir_and_outside(world):
    with pytest.raises(S.SubmitError, match='contract_candidate_002'):
        S.write_new(os.path.join(world['root'], 'contract_candidate_002', 'x'), 'a', world['root'])
    with pytest.raises(S.SubmitError, match='outside'):
        S.write_new(os.path.join(str(world['tmp']), 'elsewhere'), 'a', world['root'])


def test_job_guard_executes_and_exits_97(world, monkeypatch):
    """Run a synthetic `script` job locally (no LSF): it runs when hashes match and exits 97 before Python when a
    guarded file changes."""
    own = os.path.join(str(world['tmp']), 'own_script.py')
    open(own, 'w').write('print("RAN")\n')
    monkeypatch.setitem(S.SCRIPTS, 'synthetic', own)
    st = S._step('synthetic', S.RUN_STAGE, 'script', 1024, 10, i1=[], own_scripts=['synthetic'], args=[own])
    S.prepare(st, world['root'], {})
    job = os.path.join(world['root'], S.RUN_STAGE, 'jobs', 'synthetic', 'job.sh')
    env = {'PATH': os.environ['PATH'], 'LSB_JOBID': '77', 'LSB_JOBINDEX': '0', 'HOME': os.environ.get('HOME', '')}
    r = subprocess.run(['bash', job], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == 'RAN', r.stderr
    sc = json.load(open(os.path.join(world['root'], S.RUN_STAGE, 'host_sidecar', 'v2m_synthetic.77_0.json')))
    assert sc['hostname'] and 'cpu_model_name' in sc and sc['lsb_jobid'] == '77'
    open(own, 'a').write('# edited\n')
    env['LSB_JOBID'] = '78'
    r = subprocess.run(['bash', job], capture_output=True, text=True, env=env)
    assert r.returncode == 97 and 'RAN' not in r.stdout and 'V2M_GUARD' in r.stderr
    assert not os.path.exists(os.path.join(world['root'], S.RUN_STAGE, 'host_sidecar', 'v2m_synthetic.78_0.json'))
    open(os.path.join(world['root'], S.RUN_STAGE, 'jobs', 'synthetic', 'launch.sha256'), 'a').write('')
    os.chmod(job, 0o644)


def test_bsub_argv_rules():
    with pytest.raises(S.SubmitError, match='e-mail'):
        S.check_bsub_argv(['bsub', '-N', '-oo', 'a', '-eo', 'b', 'x'])
    with pytest.raises(S.SubmitError, match='e-mail'):
        S.check_bsub_argv(['bsub', '-u', 'me', '-oo', 'a', '-eo', 'b', 'x'])
    with pytest.raises(S.SubmitError, match='-o/-oo'):
        S.check_bsub_argv(['bsub', '-q', 'standard', 'x'])
    with pytest.raises(S.SubmitError, match='GPU'):
        S.check_bsub_argv(['bsub', '-q', 'gpu_short', '-oo', 'a', '-eo', 'b', 'x'])
    for jid, name in (('326997606', 'x'), ('5', 'a_DD_R1_b')):
        with pytest.raises(S.SubmitError, match='protected'):
            S.guard_job(jid, name)


def test_candidate_queues():
    bq = S.parse_bqueues(BQ)
    assert S.candidate_queues(360, bq) == ['standard', 'priority', 'heavy_io', 'large_mem', 'large_core_count']
    assert S.candidate_queues(30, bq)[-1] == 'short'
    assert 'rhel9_standard' not in S.candidate_queues(60, bq)
    assert 'rhel9_standard' in S.candidate_queues(60, bq, allow_rhel9=True)
    assert 'gpu' not in S.candidate_queues(30, bq, include=['gpu', 'standard'])


def test_rhel9_flag(tmp_path):
    p = str(tmp_path / 'flag.json')
    json.dump({'lead_set': True, 'ot_bitwise': True, 'bob_bitwise': False, 'os': 'RHEL 9.6'}, open(p, 'w'))
    with pytest.raises(S.SubmitError):
        S.rhel9_allowed(p)
    json.dump({'lead_set': True, 'ot_bitwise': True, 'bob_bitwise': True, 'os': 'RHEL 9.6'}, open(p, 'w'))
    assert S.rhel9_allowed(p) and not S.rhel9_allowed(None)


def test_probe_wave_records_and_split(world):
    w = S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'm3_ot_canary_w01', 40960, 360)
    wdir = os.path.join(world['root'], S.RUN_STAGE, 'queue_probe', 'm3_ot_canary_w01')
    assert open(os.path.join(wdir, 'bqueues.txt')).read() == BQ
    assert w['dispatched'] == ['standard', 'priority', 'heavy_io', 'large_core_count']    # large_mem stuck
    tsv = open(os.path.join(wdir, 'probes.tsv')).read().splitlines()
    assert len(tsv) == 6 and tsv[0].startswith('queue\tjob_id')
    probes = [c for c in world['fake'].calls if c[0] == 'bsub']
    for c in probes:
        assert c[c.index('-M') + 1] == '40960' and c[c.index('-W') + 1] == '360' and 'rusage[mem=40960]' in \
            c[c.index('-R') + 1] and c[-1] == 'hostname; date; sleep 30' and '-oo' in c and '-eo' in c
    assert world['fake'].t >= S.PROBE_WINDOW_S                   # waited the 15-min window for the stuck queue
    assert all(e['LSB_JOB_REPORT_MAIL'] == 'N' for e in world['fake'].envs)
    with pytest.raises(S.SubmitError, match='exists'):
        S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'm3_ot_canary_w01', 40960, 360)


def _prepared(world, name):
    i1 = fake_i1(world['tmp'])
    S.prepare(world['steps'][name], world['root'], i1)


def test_submit_held_readback_release_no_duplicates(world):
    _prepared(world, 'm3_ot_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    subs = S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01')
    assert [s['indices'] for s in subs] == [[1349], [1682]]
    assert [s['queue'] for s in subs] == ['standard', 'priority']
    bs = [c for c in world['fake'].calls if c[0] == 'bsub' and '-H' in c]
    assert [c[c.index('-J') + 1] for c in bs] == ['v2m_m3_ot_canary[1349]', 'v2m_m3_ot_canary[1682]']
    for c in bs:
        assert c[c.index('-oo') + 1].endswith('lsf/v2m_m3_ot_canary.%J_%I.out')
        assert c[c.index('-eo') + 1].endswith('lsf/v2m_m3_ot_canary.%J_%I.err')
    assert [c[0] for c in world['fake'].calls].count('bresume') == 2
    ev = [r['event'] for r in S.read_journal(world['root'], S.RUN_STAGE)]
    assert ev.count('submit') == 2 and ev.count('readback_ok') == 2 and ev.count('released') == 2
    n = len(world['fake'].calls)
    assert S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01') == []
    assert len(world['fake'].calls) == n


def test_submit_rejects_shape_mismatch(world):
    _prepared(world, 'm3_bob_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w02', 40960, 360)
    with pytest.raises(S.SubmitError, match='M0.4.2'):
        S.submit(world['lsf'], world['root'], world['steps']['m3_bob_canary'], 'w02')


def test_readback_failure_leaves_held(world, monkeypatch):
    _prepared(world, 'm1_bob_probe')
    S.probe_wave(world['lsf'], world['root'], 'fidelity_001', 'w01', 16384, 60)
    orig = world['fake'].__call__

    def tamper(argv, env):
        rc, out, err = orig(argv, env)
        if argv[0] == 'bsub' and '-H' in argv:
            jid = re.search(r'<(\d+)>', out).group(1)
            world['fake'].jobs[jid]['queue'] = 'other'
        return rc, out, err
    world['lsf'].runner = tamper
    with pytest.raises(S.SubmitError, match='readback'):
        S.submit(world['lsf'], world['root'], world['steps']['m1_bob_probe'], 'w01')
    assert not any(c[0] == 'bresume' for c in world['fake'].calls)


def test_split_and_index_spec():
    assert S.index_spec([1, 2, 3, 5, 7, 8]) == '1-3,5,7-8'
    idx = [i for i in range(1, 2922) if i not in (1349, 1682)]
    plan = S.split_across(idx, ['a', 'b'], 1000)
    assert sorted(i for _, c in plan for i in c) == idx and max(len(c) for _, c in plan) <= 1000
    assert [q for q, _ in plan] == ['a', 'a', 'b', 'b']
    assert S.index_spec(plan[1][1]).count(',') == 1        # 1001-1348,1350-1460-ish: canary gap respected


def _canary_out(sdir, jobname, jid, idx, mem, ok=True):
    os.makedirs(os.path.join(sdir, 'lsf'), exist_ok=True)
    status = 'Successfully completed.' if ok else 'TERM_MEMLIMIT: job killed\nExited with exit code 130.'
    open(os.path.join(sdir, 'lsf', f'{jobname}.{jid}_{idx}.out'), 'w').write(
        f'row\nSender: LSF System <x>\nSubject: Job {jid}[{idx}]: <{jobname}[{idx}]> Done\n'
        f'Job was executed on host(s) <nodeX>, in queue <standard>, as user\n{status}\n'
        f'    Max Memory :   {mem} MB\n    Run time :   100 sec.\n')


@pytest.mark.parametrize('mems,expect', [((21000, 24638), 32768), ((24639, 20000), 40960)])
def test_ot_memory_policy(world, mems, expect):
    sdir = os.path.join(world['root'], S.RUN_STAGE)
    ids = [str(10000 + i) for i in range(2921)]
    tdir = os.path.join(sdir, 'similarity', 'virchow2_fp32_ot')
    os.makedirs(tdir)
    for i, m in zip(S.OT_CANARY, mems):
        _canary_out(sdir, 'v2m_m3_ot_canary', 600 + i, i, m)
    with pytest.raises(S.SubmitError, match='rows missing'):
        S.ot_memory_decision(world['root'], ids=ids)
    for i in S.OT_CANARY:
        open(os.path.join(tdir, f'similarity_{ids[i - 1]}.pkl'), 'w').write('')
    rec = S.ot_memory_decision(world['root'], ids=ids)
    assert rec['fanout_mem_mb'] == expect
    assert S.resolve_mem(world['steps']['m3_ot_fanout'], world['root']) == expect


def test_ot_memory_requires_success(world):
    sdir = os.path.join(world['root'], S.RUN_STAGE)
    _canary_out(sdir, 'v2m_m3_ot_canary', 601, 1349, 20000, ok=False)
    _canary_out(sdir, 'v2m_m3_ot_canary', 602, 1682, 20000)
    with pytest.raises(S.SubmitError, match='not successfully completed'):
        S.ot_memory_decision(world['root'], ids=[str(i) for i in range(2921)])


def _run_canary_then_fail(world, term='TERM_MEMLIMIT'):
    _prepared(world, 'm3_ot_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    subs = S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01')
    sdir = os.path.join(world['root'], S.RUN_STAGE)
    jid = subs[0]['job_id']
    status = f'{term}: x\nExited with exit code 130.' if term else 'Exited with exit code 1.'
    open(os.path.join(sdir, 'lsf', f'v2m_m3_ot_canary.{jid}_1349.out'), 'w').write(
        f'Sender: LSF System\nSubject: Job {jid}[1349]: <v2m_m3_ot_canary[1349]> Exited\n{status}\n')
    world['fake'].jobs[jid]['el'][1349] = 'EXIT'          # F2: LSF reports the element EXIT
    return subs


def _exit_with_out(world, jid, index=1349, term='TERM_MEMLIMIT'):
    sdir = os.path.join(world['root'], S.RUN_STAGE)
    open(os.path.join(sdir, 'lsf', f'v2m_m3_ot_canary.{jid}_{index}.out'), 'w').write(
        f'Sender: LSF System\nSubject: Job {jid}[{index}]: Exited\n{term}: x\nExited with exit code 130.\n')
    world['fake'].jobs[jid]['el'][index] = 'EXIT'


def test_retry_once_memlimit(world):
    _run_canary_then_fail(world)
    with pytest.raises(S.SubmitError, match='retry-shape'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'w01')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    rec = S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    c = rec['argv']
    assert c[c.index('-M') + 1] == '65536' and c[c.index('-W') + 1] == '720'
    assert c[c.index('-J') + 1] == 'v2m_m3_ot_canary[1349]' and '-H' in c
    tsv = open(os.path.join(world['root'], S.RUN_STAGE, 'resubmissions.tsv')).read().splitlines()
    assert len(tsv) == 2 and 'TERM_MEMLIMIT' in tsv[1]
    _exit_with_out(world, rec['job_id'])             # F2: the retry element itself must be EXIT with its out file
    with pytest.raises(S.SubmitError, match='already retried'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    failed = open(os.path.join(world['root'], S.RUN_STAGE, 'failed_elements.jsonl')).read()
    assert 'second failure' in failed


def test_retry_ineligible_marks_failed(world):
    _run_canary_then_fail(world, term=None)
    world['fake'].bhist_text = 'Job <501>, ...\n Exited with exit code 1.\n'
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    with pytest.raises(S.SubmitError, match='package FAILED'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    assert 'not eligible' in open(os.path.join(world['root'], S.RUN_STAGE, 'failed_elements.jsonl')).read()


def test_retry_host_failure_quoted(world):
    _run_canary_then_fail(world, term=None)
    world['fake'].bhist_text = 'Mon Oct 13 22:00:00: Exited; host <noderome5> failure, job requeue not set;\n'
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    rec = S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    assert rec['reason'] == 'host_failure' and 'noderome5' in rec['bhist_line']


def test_migrate_and_monitor_switch_only_pend(world):
    _prepared(world, 'm3_bob_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 16384, 480)
    subs = S.submit(world['lsf'], world['root'], world['steps']['m3_bob_canary'], 'w01')
    jid = subs[0]['job_id']
    log = os.path.join(world['root'], S.RUN_STAGE, 'monitor.log')
    with pytest.raises(S.SubmitError, match='did not dispatch'):
        S.migrate(world['lsf'], world['root'], S.RUN_STAGE, 'm3_bob_canary', 'large_mem', 'w01')
    out = S.migrate(world['lsf'], world['root'], S.RUN_STAGE, 'm3_bob_canary', 'heavy_io', 'w01')
    assert len(out) == 1 and '\tSWITCH\t' + jid + '\t68\tstandard\theavy_io' in out[0]
    assert out[0] in open(log).read()
    world['fake'].jobs[jid]['el'][68] = 'RUN'
    assert S.migrate(world['lsf'], world['root'], S.RUN_STAGE, 'm3_bob_canary', 'priority', 'w01') == []
    with pytest.raises(S.SubmitError, match='not a journaled element'):
        S.switch_element(world['lsf'], world['root'], S.RUN_STAGE, '326997606', 0, 'standard', 'w01', log, 'x')
    probe_kills = {r['job_id'] for r in S.read_journal(world['root'], S.RUN_STAGE) if r['event'] == 'probe_bkill'}
    assert not any(c[0] in ('bstop', 'bmod') for c in world['fake'].calls)
    assert all(c[1] in probe_kills for c in world['fake'].calls if c[0] == 'bkill')     # F12: own probes only
    assert not any('326997606' in ' '.join(c) or '900' == c[-1] for c in world['fake'].calls if c[0] == 'bswitch')


def test_monitor_observe_only_by_default_and_auto_switch(world):
    _prepared(world, 'm3_bob_fanout')
    # fan-out requires the BoB canary rows; satisfy the precondition synthetically
    sdir = os.path.join(world['root'], S.RUN_STAGE)
    ids = [str(10000 + i) for i in range(2921)]
    _canary_out(sdir, 'v2m_m3_bob_canary', 700, 68, 5000)
    tdir = os.path.join(sdir, 'similarity', 'virchow2_fp32_bob')
    os.makedirs(tdir)
    for r in range(1341, 1361):
        open(os.path.join(tdir, f'similarity_{ids[r - 1]}.pkl'), 'w').write('')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 16384, 480)
    subs = S.submit(world['lsf'], world['root'], world['steps']['m3_bob_fanout'], 'w01', ids=ids)
    assert sorted(i for s in subs for i in s['indices']) == [b for b in range(1, 148) if b != 68]
    n_calls = len(world['fake'].calls)
    summ, state, pend = S.monitor_once(world['lsf'], world['root'], S.RUN_STAGE)
    assert summ['states'] == {'PEND': 146} and summ['switched'] == 0
    assert not any(c[0] == 'bswitch' for c in world['fake'].calls[n_calls:])
    log = open(os.path.join(sdir, 'monitor.log')).read()
    assert log.count('\tSTATE\t') == 146 and '\tSUMMARY\t' in log
    first = subs[0]
    world['fake'].jobs[first['job_id']]['el'][first['indices'][0]] = 'RUN'
    world['fake'].t += 10 ** 10       # make every element "old"
    summ, state, pend = S.monitor_once(world['lsf'], world['root'], S.RUN_STAGE, switch_after_min=60, state=state)
    sw = [c for c in world['fake'].calls if c[0] == 'bswitch']
    assert sw and all(re.match(r'\d+\[\d+\]$', c[2]) for c in sw)
    assert first['job_id'] + f'[{first["indices"][0]}]' not in [c[2] for c in sw]       # RUN never switched
    assert all(c[1] in ('standard', 'priority', 'heavy_io', 'large_core_count') for c in sw)


def test_dry_run_calls_no_lsf(world, capsys):
    fake = world['fake']
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'wd', 40960, 360, dry_run=True)
    S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'wd', dry_run=True)
    assert fake.calls == [] and not os.path.exists(os.path.join(world['root'], S.RUN_STAGE))
    out = capsys.readouterr().out
    assert 'DRY probe' in out and 'v2m_m3_ot_canary[1349]' in out and '-H' in out


def test_jobfile_submission(world):
    rd = os.path.join(world['root'], S.RUN_STAGE)
    os.makedirs(rd)
    jf = os.path.join(rd, 'eval_job_v2main_001.lsf')
    open(jf, 'w').write('#!/bin/bash\n#BSUB -J r21_v2main_fp32_eval\n#BSUB -q heavy_io\n#BSUB -oo a\n#BSUB -eo b\n')
    st = world['steps']['m4_evaluator']
    st['jobfile'] = jf
    S.prepare(st, world['root'])
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w05', 8192, 45, include=['heavy_io', 'standard'])
    seen = []
    orig = world['fake'].__call__

    def runner(argv, env):
        seen.append(argv)
        if argv[:2] == ['bash', '-c']:
            return orig(['bsub', '-H', '-q', 'heavy_io', '-J', 'r21_v2main_fp32_eval', '-oo', 'a', '-eo', 'b', jf], env)
        return orig(argv, env)
    world['lsf'].runner = runner
    subs = S.submit(world['lsf'], world['root'], st, 'w05')
    assert subs[0]['queue'] == 'heavy_io' and any(a[:2] == ['bash', '-c'] and 'bsub -H <' in a[2] for a in seen)
    open(jf, 'a').write('#BSUB -N\n')
    with pytest.raises(S.SubmitError):
        S.submit(world['lsf'], world['root'], st, 'w05')


# ---------------------------------------------------------------------------------------------- repair 002

def _snapshot(world):
    sdir = os.path.join(world['root'], S.RUN_STAGE)
    return {f: open(os.path.join(sdir, f)).read() for f in ('journal.jsonl', 'failed_elements.jsonl',
                                                             'resubmissions.tsv') if os.path.exists(os.path.join(sdir, f))}


@pytest.mark.parametrize('stat', ['RUN', 'PEND', 'PSUSP', 'USUSP', 'SSUSP', 'UNKWN', 'ZOMBI'])
def test_f2_live_element_no_out_file_stops_without_writing(world, stat):
    """Review probe P2: element still alive, no LSF out file -> stop, nothing written, never FAILED."""
    subs = _run_canary_then_fail(world, term=None)
    jid = subs[0]['job_id']
    os.remove(os.path.join(world['root'], S.RUN_STAGE, 'lsf', f'v2m_m3_ot_canary.{jid}_1349.out'))
    world['fake'].jobs[jid]['el'][1349] = stat
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    before, n = _snapshot(world), len([c for c in world['fake'].calls if c[0] == 'bsub'])
    with pytest.raises(S.SubmitError, match=f'is {stat}, not EXIT/DONE'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    assert _snapshot(world) == before and 'failed_elements.jsonl' not in before
    assert len([c for c in world['fake'].calls if c[0] == 'bsub']) == n
    assert not any(c[0] == 'bhist' for c in world['fake'].calls)


def test_f2_requeue_host_line_on_running_element_no_duplicate(world):
    """Review probe P2 variant: bhist of an LSF-requeued, now RUN element mentions 'host <x> closed'."""
    subs = _run_canary_then_fail(world, term=None)
    jid = subs[0]['job_id']
    world['fake'].jobs[jid]['el'][1349] = 'RUN'
    world['fake'].bhist_text = 'Mon Oct 13 22:00:00: Job requeued; host <noderome5> closed;\n'
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    before, n = _snapshot(world), len([c for c in world['fake'].calls if c[0] == 'bsub'])
    with pytest.raises(S.SubmitError, match='is RUN'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    assert _snapshot(world) == before
    assert len([c for c in world['fake'].calls if c[0] == 'bsub']) == n


def test_f2_exit_without_out_file_and_unknown_job_stop(world):
    subs = _run_canary_then_fail(world)
    jid = subs[0]['job_id']
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    out = os.path.join(world['root'], S.RUN_STAGE, 'lsf', f'v2m_m3_ot_canary.{jid}_1349.out')
    os.rename(out, out + '.x')
    before = _snapshot(world)
    with pytest.raises(S.SubmitError, match='output file is missing'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    os.rename(out + '.x', out)
    del world['fake'].jobs[jid]                       # bjobs no longer knows the job (CLEAN_PERIOD)
    with pytest.raises(S.SubmitError, match='state unknown'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    assert _snapshot(world) == before


def test_f2_done_element_no_retry(world):
    subs = _run_canary_then_fail(world)
    world['fake'].jobs[subs[0]['job_id']]['el'][1349] = 'DONE'
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    before = _snapshot(world)
    with pytest.raises(S.SubmitError, match='completed successfully'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    assert _snapshot(world) == before


def test_f2_second_attempt_running_not_marked_failed(world):
    _run_canary_then_fail(world)
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    rec = S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    world['fake'].jobs[rec['job_id']]['el'][1349] = 'RUN'
    with pytest.raises(S.SubmitError, match='is RUN'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    assert not os.path.exists(os.path.join(world['root'], S.RUN_STAGE, 'failed_elements.jsonl'))


def test_f2_host_regex_final_dispatch_only(world):
    assert not S.HOST_FAIL_RE.search('Mon Oct 13 22:00:00: Job requeued; host <noderome5> closed;')
    assert not S.HOST_FAIL_RE.search('Mon Oct 13 22:00:00: Dispatched 1 Task(s) on Host(s) <nodeA>, Allocated 1 '
                                     'Slot(s) on Host(s) <nodeA>, Effective RES_REQ <span[hosts=1] >;')
    assert S.HOST_FAIL_RE.search('Mon Oct 13 22:00:00: Exited; host <noderome5> failure, job requeue not set;')
    assert S.HOST_FAIL_RE.search('Mon Oct 13 22:00:00: Exited by LSF signal TERM_HOST;')
    hist = ('Mon Oct 13 20:00:00: Dispatched 1 Task(s) on Host(s) <n1>;\n'
            'Mon Oct 13 20:10:00: Job requeued; host <n1> unreachable;\n'
            'Mon Oct 13 20:20:00: Dispatched 1 Task(s) on Host(s) <n2>;\n'
            'Mon Oct 13 21:00:00: Exited with exit code 1.\n')
    assert not any(S.HOST_FAIL_RE.search(x) for x in S.final_dispatch_lines(hist))
    subs = _run_canary_then_fail(world, term=None)
    world['fake'].bhist_text = hist
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    with pytest.raises(S.SubmitError, match='package FAILED'):
        S.retry(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 1349, 'r01')
    assert 'rc=1' in open(os.path.join(world['root'], S.RUN_STAGE, 'failed_elements.jsonl')).read()


def test_f11_release_subcommand(world):
    """First held job fails readback; the run is then interrupted before the second readback (F17: submit itself now
    reads back every chunk, so an un-read-back held job only arises from an interrupted run) and the later held job
    is left PSUSP. `release` releases the clean one and refuses the one whose readback failed."""
    _prepared(world, 'm3_ot_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    orig = world['fake'].__call__
    state = {'n': 0}

    def tamper(argv, env):
        rc, out, err = orig(argv, env)
        if argv[0] == 'bsub' and '-H' in argv:
            state['n'] += 1
            if state['n'] == 1:
                world['fake'].jobs[re.search(r'<(\d+)>', out).group(1)]['queue'] = 'other'
        return rc, out, err
    world['lsf'].runner = tamper
    real, n_rb = S.readback_and_release, {'n': 0}

    def interrupted(*a, **k):
        n_rb['n'] += 1
        if n_rb['n'] == 2:
            raise KeyboardInterrupt('simulated interruption before the second readback')
        return real(*a, **k)
    S.readback_and_release = interrupted
    try:
        with pytest.raises(KeyboardInterrupt):
            S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01')
    finally:
        S.readback_and_release = real
    jr = S.read_journal(world['root'], S.RUN_STAGE)
    bad, good = [r['job_id'] for r in jr if r['event'] == 'submit']
    assert world['fake'].jobs[good]['el'][1682] == 'PSUSP'
    world['lsf'].runner = orig
    S.release(world['lsf'], world['root'], S.RUN_STAGE, dry_run=True)
    assert world['fake'].jobs[good]['el'][1682] == 'PSUSP'
    with pytest.raises(S.SubmitError, match=f'refused.*{bad}'):
        S.release(world['lsf'], world['root'], S.RUN_STAGE)
    assert world['fake'].jobs[good]['el'][1682] == 'PEND' and world['fake'].jobs[bad]['el'][1349] == 'PSUSP'
    resumed = [c[1] for c in world['fake'].calls if c[0] == 'bresume']
    assert resumed == [good]
    assert os.path.exists(os.path.join(world['root'], S.RUN_STAGE, 'jobs', 'm3_ot_canary',
                                       f'readback_{good}_release001.json'))
    with pytest.raises(S.SubmitError, match='refused'):          # nothing left but the refused job
        S.release(world['lsf'], world['root'], S.RUN_STAGE)
    assert [c[1] for c in world['fake'].calls if c[0] == 'bresume'] == [good]
    assert S.main(['--root', world['root'], 'release', '--stage', S.RUN_STAGE]) == 1


def test_f12_probe_wave_kills_only_own_undispatched_probes(world):
    w = S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    kills = [c for c in world['fake'].calls if c[0] == 'bkill']
    stuck = [j for j, v in world['fake'].jobs.items() if v['name'] == 'v2m_probe_w01_large_mem'][0]
    assert kills == [['bkill', stuck]] and w['killed_undispatched_probes'] == [{'job_id': stuck, 'rc': 0}]
    jr = [r for r in S.read_journal(world['root'], S.RUN_STAGE) if r['event'] == 'probe_bkill']
    assert [r['job_id'] for r in jr] == [stuck]
    assert world['fake'].jobs['326997606']['el'][0] == 'RUN' and world['fake'].jobs['900']['el'][0] == 'RUN'


def test_f12_probe_dispatched_late_is_not_killed(world):
    orig = world['fake'].__call__
    seen = {'n': 0}

    def late(argv, env):
        if argv[0] == 'bjobs' and '-l' not in argv and len(argv[5:]) == 1:     # the fresh pre-kill read only
            seen['n'] += 1
            for j in world['fake'].jobs.values():                 # the stuck probe starts right after the window
                if j['name'] == 'v2m_probe_w01_large_mem':
                    j['el'][0] = 'RUN'
        return orig(argv, env)
    world['lsf'].runner = late
    w = S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    assert w['killed_undispatched_probes'] == [] and not any(c[0] == 'bkill' for c in world['fake'].calls)
    assert 'large_mem' not in w['dispatched'] and seen['n'] == 1


def test_f12_never_kills_foreign_ids(world):
    probes = {'gpu': {'job_id': '326997606', 'dispatch_t': None}, 'x': {'job_id': '900', 'dispatch_t': None},
              'y': {'job_id': '777', 'dispatch_t': None}}
    assert S.kill_undispatched_probes(world['lsf'], world['root'], S.RUN_STAGE, 'w01', probes, 900) == []
    assert world['fake'].calls == []


def test_f13_dry_run_m4_evaluator_missing_jobfile(world, capsys):
    st = world['steps']['m4_evaluator']
    st['jobfile'] = os.path.join(world['root'], S.RUN_STAGE, 'absent.lsf')
    with pytest.raises(S.SubmitError, match='not written yet'):
        S.submit(world['lsf'], world['root'], st, 'w05', dry_run=True)
    assert world['fake'].calls == []
    assert S.main(['--root', world['root'], '--dry-run', 'submit', '--step', 'm4_evaluator', '--wave', 'w05']) == 1
    assert 'STOP' in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- repair 003 (F17)

def _held_view(world, override):
    """Runner wrapper: `override(jid, n_query) -> stat or None` rewrites every element stat of job `jid` in a
    single-job bjobs -o query; n_query counts that job's -o queries (1-based). Returns the per-job query counter."""
    orig = world['fake'].__call__
    count = {}

    def run(argv, env):
        rc, out, err = orig(argv, env)
        if argv[0] == 'bjobs' and '-l' not in argv and len(argv[5:]) == 1:
            jid = argv[5]
            count[jid] = count.get(jid, 0) + 1
            st = override(jid, count[jid])
            if st is not None:
                lines = []
                for ln in out.splitlines():
                    f = ln.split('|')
                    if len(f) >= 9 and f[0] == jid:
                        f[2] = st
                    lines.append('|'.join(f))
                out = '\n'.join(lines) + '\n'
        return rc, out, err
    world['lsf'].runner = run
    return count


def _readback(world, stage, step, jid, suffix=''):
    return json.load(open(os.path.join(world['root'], stage, 'jobs', step, f'readback_{jid}{suffix}.json')))


def test_f17_pend_then_psusp_settles_and_records(world):
    _prepared(world, 'm3_ot_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    _held_view(world, lambda jid, n: 'PEND' if n <= 3 else None)      # 3 transient PEND polls, then PSUSP
    subs = S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01')
    assert len(subs) == 2
    for s in subs:
        rb = _readback(world, S.RUN_STAGE, 'm3_ot_canary', s['job_id'])
        st = rb['settle']
        assert rb['problems'] == [] and st['timed_out'] is False and st['n_polls'] == 4
        assert [p['rows'][0]['stat'] for p in st['polls']] == ['PEND', 'PEND', 'PEND', 'PSUSP']
        assert st['settle_s'] == pytest.approx(3 * S.READBACK_POLL_S) and st['window_s'] == S.READBACK_SETTLE_S
        assert [p['t_s'] for p in st['polls']] == [0.0, 2.5, 5.0, 7.5]
    jr = S.read_journal(world['root'], S.RUN_STAGE)
    ok = [r for r in jr if r['event'] == 'readback_ok']
    assert [r['settle_polls'] for r in ok] == [4, 4] and not any(r['event'] == 'readback_failed' for r in jr)
    assert [c[1] for c in world['fake'].calls if c[0] == 'bresume'] == [s['job_id'] for s in subs]


def test_f17_pend_beyond_window_fails_as_before(world):
    _prepared(world, 'm3_ot_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    bad = str(world['fake'].next_id)                       # id of the first held submission
    _held_view(world, lambda jid, n: 'PEND' if jid == bad else None)
    jr0 = len(S.read_journal(world['root'], S.RUN_STAGE))
    with pytest.raises(S.SubmitError, match=rf"failed for held job\(s\) \['{bad}'\]"):
        S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01')
    rb = _readback(world, S.RUN_STAGE, 'm3_ot_canary', bad)
    st = rb['settle']
    assert st['timed_out'] is True and st['settle_s'] >= S.READBACK_SETTLE_S
    assert st['n_polls'] <= int(S.READBACK_SETTLE_S / S.READBACK_POLL_S) + 2
    assert all(p['rows'][0]['stat'] == 'PEND' for p in st['polls'])
    assert any('stat PEND != PSUSP' in p for p in rb['problems']) and any('not settled' in p for p in rb['problems'])
    jr = S.read_journal(world['root'], S.RUN_STAGE)[jr0:]
    assert [r['job_id'] for r in jr if r['event'] == 'readback_failed'] == [bad]
    assert bad not in [c[1] for c in world['fake'].calls if c[0] == 'bresume']
    assert world['fake'].jobs[bad]['el'][1349] == 'PSUSP'                     # left held for the analyst
    assert len([r for r in jr if r['event'] == 'released']) == 1              # the sibling chunk was released


@pytest.mark.parametrize('stat', ['RUN', 'DONE', 'EXIT', 'USUSP', 'SSUSP'])
def test_f17_other_state_on_first_poll_fails_immediately(world, stat):
    _prepared(world, 'm3_ot_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    t0 = world['fake'].t
    count = _held_view(world, lambda jid, n: stat)
    with pytest.raises(S.SubmitError, match='readback/release failed'):
        S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01')
    subs = [r for r in S.read_journal(world['root'], S.RUN_STAGE) if r['event'] == 'submit']
    for s in subs:
        rb = _readback(world, S.RUN_STAGE, 'm3_ot_canary', s['job_id'])
        assert rb['settle']['n_polls'] == 1 and rb['settle']['timed_out'] is False
        assert rb['problems'] == [f'{s["job_id"]}[{s["indices"][0]}] stat {stat} != PSUSP']
        assert count[s['job_id']] == 1                                           # no further polling
    assert world['fake'].t == t0                                                 # never slept
    assert not any(c[0] == 'bresume' for c in world['fake'].calls)


def test_f17_mixed_pend_and_run_elements_not_waited():
    rows_seq = [[{'jobid': '7', 'index': 1, 'stat': 'PEND'}, {'jobid': '7', 'index': 2, 'stat': 'RUN'}]]

    class L:
        t = 0.0

        def clock(self):
            return self.t

        def sleep(self, s):
            self.t += s

        def bjobs_rows(self, ids):
            return rows_seq[0]
    rows, st = S.settle_held_rows(L(), '7')
    assert st['n_polls'] == 1 and st['timed_out'] is False and st['settle_s'] == 0.0


def test_f17_multichunk_one_fails_others_released(world):
    _prepared(world, 'm1_bob_probe')
    S.probe_wave(world['lsf'], world['root'], 'fidelity_001', 'w01', 16384, 60)
    orig = world['fake'].__call__
    n = {'k': 0}

    def tamper(argv, env):
        rc, out, err = orig(argv, env)
        if argv[0] == 'bsub' and '-H' in argv:
            n['k'] += 1
            if n['k'] == 3:                                                      # third chunk: wrong queue
                world['fake'].jobs[re.search(r'<(\d+)>', out).group(1)]['queue'] = 'other'
        return rc, out, err
    world['lsf'].runner = tamper
    with pytest.raises(S.SubmitError) as ei:
        S.submit(world['lsf'], world['root'], world['steps']['m1_bob_probe'], 'w01')
    jr = S.read_journal(world['root'], 'fidelity_001')
    subs = [r for r in jr if r['event'] == 'submit']
    assert len(subs) == 7
    bad = subs[2]['job_id']
    good = [s['job_id'] for s in subs if s['job_id'] != bad]
    msg = str(ei.value)
    assert f"held job(s) ['{bad}']" in msg and f'released {good}' in msg
    assert [r['job_id'] for r in jr if r['event'] == 'readback_failed'] == [bad]
    assert [r['job_id'] for r in jr if r['event'] == 'released'] == good
    assert [c[1] for c in world['fake'].calls if c[0] == 'bresume'] == good
    for s in subs:                                                               # every chunk read back
        assert os.path.exists(os.path.join(world['root'], 'fidelity_001', 'jobs', 'm1_bob_probe',
                                           f'readback_{s["job_id"]}.json'))
    nb = len([c for c in world['fake'].calls if c[0] == 'bsub'])
    assert S.submit(world['lsf'], world['root'], world['steps']['m1_bob_probe'], 'w01') == []   # once-only
    assert len([c for c in world['fake'].calls if c[0] == 'bsub']) == nb


def test_f17_bresume_failure_journaled_and_others_continue(world):
    _prepared(world, 'm3_ot_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    orig = world['fake'].__call__
    n = {'k': 0}

    def flaky(argv, env):
        if argv[0] == 'bresume':
            n['k'] += 1
            if n['k'] == 1:
                return 255, '', 'bresume: transient failure'
        return orig(argv, env)
    world['lsf'].runner = flaky
    with pytest.raises(S.SubmitError, match='readback/release failed'):
        S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01')
    jr = S.read_journal(world['root'], S.RUN_STAGE)
    subs = [r['job_id'] for r in jr if r['event'] == 'submit']
    assert [r['job_id'] for r in jr if r['event'] == 'release_failed'] == [subs[0]]
    assert [r['job_id'] for r in jr if r['event'] == 'released'] == [subs[1]]
    assert S.release(world['lsf'], world['root'], S.RUN_STAGE) == [subs[0]]       # F11 path unchanged


def test_f17_unknown_lead_events_tolerated(world):
    _prepared(world, 'm3_ot_canary')
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'w01', 40960, 360)
    _held_view(world, lambda jid, n: 'PEND')                                    # every readback times out
    with pytest.raises(S.SubmitError, match='readback/release failed'):
        S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01')
    world['lsf'].runner = world['fake']
    subs = [r for r in S.read_journal(world['root'], S.RUN_STAGE) if r['event'] == 'submit']
    jp = os.path.join(world['root'], S.RUN_STAGE, 'journal.jsonl')
    with open(jp, 'a') as fh:                                                    # analyst-authored, outside the wrapper
        for s in subs:
            world['fake'].jobs[s['job_id']]['el'] = {i: 'PEND' for i in s['indices']}
            fh.write(json.dumps({'event': 'lead_manual_release', 'job_id': s['job_id'], 'step': s['step'],
                                 'lead_record': 'lead_logs/lead_manual_readback_001.json',
                                 'time': '2026-10-14T01:40:00-05:00'}) + '\n')
        fh.write(json.dumps({'event': 'lead_note', 'text': 'no job_id/step keys'}) + '\n')
        fh.write(json.dumps({'note': 'no event key at all'}) + '\n')
    n_calls = len(world['fake'].calls)
    jr = S.read_journal(world['root'], S.RUN_STAGE)
    assert [r.get('event') for r in jr[-3:]] == ['lead_manual_release', 'lead_note', None]
    assert sorted(S.journaled_elements(world['root'], S.RUN_STAGE, 'm3_ot_canary')) == [1349, 1682]
    assert len(S.our_elements(world['root'], S.RUN_STAGE)) == 2
    with pytest.raises(S.SubmitError, match='refused'):                          # readback_failed stays refused
        S.release(world['lsf'], world['root'], S.RUN_STAGE)
    assert not any(c[0] == 'bresume' for c in world['fake'].calls[n_calls:])
    summ, state, pend = S.monitor_once(world['lsf'], world['root'], S.RUN_STAGE)
    assert summ['n_elements'] == 2 and summ['states'] == {'PEND': 2} and summ['switched'] == 0
    assert S.submit(world['lsf'], world['root'], world['steps']['m3_ot_canary'], 'w01') == []
    assert not any(c[0] in ('bsub', 'bresume', 'bkill', 'bswitch') for c in world['fake'].calls[n_calls:])


# ----------------------------------------------------------------------------------------------
# AMENDMENT_003: run_001 consumers of the fidelity directory read fidelity_002
# ----------------------------------------------------------------------------------------------

def test_a003_run_001_fidelity_consumers_use_fidelity_002(world):
    st, root = world['steps'], world['root']
    fd1, fd2 = os.path.join(root, 'fidelity_001'), os.path.join(root, 'fidelity_002')
    assert S.FIDELITY_STAGE_A003 == 'fidelity_002'
    a = st['m3_gates']['args']
    assert a[1] == 'm3' and a[a.index('--fidelity-dir') + 1] == fd2 and st['m3_gates']['stage'] == S.RUN_STAGE
    for m in S.METHODS:
        s = st[f'm1_ranks_{m}']
        # review F4: the m1_ranks_* CLI writes into fidelity_001, so its precheck inspects fidelity_001 too
        assert s['pre'] == [[S.PY, S.SCRIPTS['fidelity'], 'ranks-precheck', '--stage-dir', fd1]]
        assert s['stage'] == 'fidelity_001'                          # DONE fidelity-stage steps keep 001
        assert S.stage_dir(root, s['stage']) == fd1
    for k, s in st.items():                                          # no run_001 step references fidelity_001
        if s['stage'] == S.RUN_STAGE:
            flat = [x for x in s['args']] + [x for p in s.get('pre', []) for x in p]
            assert not any(isinstance(x, str) and fd1 in x for x in flat), k
    fid = [k for k, s in st.items() if s['stage'] == 'fidelity_001']
    assert fid and all(not any(isinstance(x, str) and fd2 in x for x in st[k]['args'] +
                               [y for p in st[k].get('pre', []) for y in p]) for k in fid)
    users = sorted(k for k, s in st.items() if any(isinstance(x, str) and fd2 in x for x in
                                                   s['args'] + [y for p in s.get('pre', []) for y in p]))
    assert users == ['m3_gates']                                     # the only submitted fidelity_002 consumer
    with pytest.raises(S.SubmitError, match='unknown stage'):
        S.stage_dir(root, 'fidelity_002')                            # submit can never target fidelity_002


# ----------------------------------------------------------------------------------------------
# AMENDMENT_004 / ADDENDUM_003: every run-stage step resolves to RUN_STAGE (run_003; run_001 and run_002 are
# frozen FAILED). implementation_008 adapted these tests deliberately from run_002 to run_003.
# ----------------------------------------------------------------------------------------------

def _flat(s):
    return [x for x in s['args'] + [y for p in s.get('pre', []) for y in p] + (s['params'] or []) +
            [s['jobfile'] or ''] if isinstance(x, str)]


def test_a004_run_stage_resolution_for_every_run_step(world):
    st, root = world['steps'], world['root']
    rd1, rd2 = os.path.join(root, 'run_001'), os.path.join(root, 'run_003')
    old2 = os.path.join(root, 'run_002')
    assert S.RUN_STAGE == 'run_003' and S.STAGES == ('fidelity_001', 'run_003')
    assert S.stage_dir(root, S.RUN_STAGE) == rd2
    for old in ('run_001', 'run_002'):
        with pytest.raises(S.SubmitError, match=f'unknown stage {old}'):
            S.stage_dir(root, old)                                   # frozen FAILED stages are never targeted
    run_steps = sorted(k for k, s in st.items() if s['stage'] != 'fidelity_001')
    assert run_steps == sorted(
        [f'm3_aggregate_{m}' for m in ('median_pooling', 'sampler')] + ['m3_combine_ot', 'm3_combine_bob'] +
        [f'm4_ranks_{m}' for m in S.METHODS] + ['m3_ot_canary', 'm3_ot_fanout', 'm3_bob_canary', 'm3_bob_fanout',
                                                'm3_gates', 'm0_input_bindings_run', 'm4_ranks_gate',
                                                'm5_tie_hosts', 'm4_evaluator', 'm4_adapter_run_a', 'm5_compare',
                                                'm4_adapter_run_b'])           # ADDENDUM_005 ruling 1
    for k in run_steps:
        s = st[k]
        assert s['stage'] == S.RUN_STAGE, k
        assert not any(rd1 in x or old2 in x for x in _flat(s)), k   # nothing under run_001/run_002 is referenced
        if '--run-dir' in s['args']:
            assert s['args'][s['args'].index('--run-dir') + 1] == rd2, k
    assert st['m4_evaluator']['jobfile'] == os.path.join(rd2, 'eval_job_v2main_001.lsf')
    assert all(os.path.join(rd2, 'similarity') in st[f'm3_combine_{m}']['args'][
        st[f'm3_combine_{m}']['args'].index('--similarity-dir') + 1] for m in ('ot', 'bob'))
    a = st['m0_input_bindings_run']['args']
    assert a[a.index('--stage') + 1] == 'run_003' and a[a.index('--stage-dir') + 1] == rd2
    a = st['m3_gates']['args']
    assert a[a.index('--amendment') + 1] == os.path.join(root, 'contract_amendment_004', 'AMENDMENT_004.md')
    assert a[a.index('--run-dir') + 1] == rd2


def test_a004_prepare_writes_jobs_under_run_stage(world):
    st, root = world['steps'], world['root']
    for k in ('m3_gates', 'm4_ranks_gate', 'm5_tie_hosts', 'm0_input_bindings_run'):
        S.prepare(st[k], root, {})
        assert os.path.isfile(os.path.join(root, 'run_003', 'jobs', k, 'job.sh')), k
        job = open(os.path.join(root, 'run_003', 'jobs', k, 'job.sh')).read()
        assert os.path.join(root, 'run_001') not in job and os.path.join(root, 'run_002') not in job, k
        S.verify_prepared(st[k], root)
    assert not os.path.lexists(os.path.join(root, 'run_001')) and not os.path.lexists(os.path.join(root, 'run_002'))


def test_a004_inherited_run_001_m3_gates_record_not_counted(world):
    """ADDENDUM_003: the run_003 journal inherits run_001 AND run_002 records; the retried steps (m3_gates,
    m4_ranks_gate) count only RUN_STAGE submissions, every other step still counts its inherited records."""
    st, root = world['steps'], world['root']
    rd1, rd2, rd3 = (os.path.join(root, r) for r in ('run_001', 'run_002', 'run_003'))
    assert S.RUN_STAGE_RETRIED_STEPS == ('m3_gates', 'm4_ranks_gate')
    os.makedirs(rd3)
    inherited = [{'event': 'submit', 'step': 'm3_gates', 'job_id': '332494769', 'indices': [0], 'held': True,
                  'argv': ['bsub', '-oo', os.path.join(rd1, 'lsf', 'v2m_m3_gates.%J_%I.out'),
                           os.path.join(rd1, 'jobs', 'm3_gates', 'job.sh')]},
                 {'event': 'submit', 'step': 'm3_ot_fanout', 'job_id': '900', 'indices': [1, 2, 3],
                  'argv': ['bsub', os.path.join(rd1, 'jobs', 'm3_ot_fanout', 'job.sh')]},
                 {'event': 'submit', 'step': 'm3_gates', 'job_id': '332520983', 'indices': [0],
                  'argv': ['bsub', os.path.join(rd2, 'jobs', 'm3_gates', 'job.sh')]},
                 {'event': 'submit', 'step': 'm4_ranks_sampler', 'job_id': '333', 'indices': [0],
                  'argv': ['bsub', os.path.join(rd2, 'jobs', 'm4_ranks_sampler', 'job.sh')]},
                 {'event': 'submit', 'step': 'm4_ranks_gate', 'job_id': '334', 'indices': [0],
                  'argv': ['bsub', os.path.join(rd2, 'jobs', 'm4_ranks_gate', 'job.sh')]}]
    open(os.path.join(rd3, 'journal.jsonl'), 'w').write(''.join(json.dumps(r) + '\n' for r in inherited))
    assert S.journaled_elements(root, S.RUN_STAGE, 'm3_gates') == {}
    assert S.journaled_elements(root, S.RUN_STAGE, 'm4_ranks_gate') == {}
    assert sorted(S.journaled_elements(root, S.RUN_STAGE, 'm3_ot_fanout')) == [1, 2, 3]   # fan-outs never re-run
    assert sorted(S.journaled_elements(root, S.RUN_STAGE, 'm4_ranks_sampler')) == [0]     # run_002 ranks kept
    for step in ('m4_ranks_gate', 'm3_gates'):
        S.prepare(st[step], root, {})
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, 'w01', 16384, 60)
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, 'w02', st['m4_ranks_gate']['mem_mb'],
                 st['m4_ranks_gate']['wall_min'])
    for step, wave in (('m4_ranks_gate', 'w02'), ('m3_gates', 'w01')):
        subs = S.submit(world['lsf'], root, st[step], wave)
        assert len(subs) == 1 and any(rd3 in x for x in subs[0]['argv'])
        assert not any(rd1 in x or rd2 in x for x in subs[0]['argv'])
        assert [r['job_id'] for r in S.journaled_elements(root, S.RUN_STAGE, step)[0]] == [subs[0]['job_id']]
        n = len(world['fake'].calls)
        assert S.submit(world['lsf'], root, st[step], wave) == []      # the run_003 retry is submitted once
        assert not any(c[0] == 'bsub' for c in world['fake'].calls[n:])
    assert not os.path.lexists(rd1) and not os.path.lexists(rd2)


def test_a8_wave_carried_from_earlier_stage_refused(world):
    """ADDENDUM_003 / M0.4.2 (a wave per stage): run_002 waves carried into run_003 by lineage are never used."""
    st, root = world['steps'], world['root']
    rd3 = os.path.join(root, 'run_003')
    step = st['m4_ranks_gate']
    for wave, stage in (('w4096_30_r002', 'run_002'), ('w16384_60', 'run_001'), ('w_nostage', None)):
        wd = os.path.join(rd3, 'queue_probe', wave)
        os.makedirs(wd)
        rec = {'wave': wave, 'kind': 'production', 'mem_mb': step['mem_mb'], 'wall_min': step['wall_min'],
               'candidates': ['standard'], 'dispatched': ['standard'], 'allow_rhel9': False}
        if stage:
            rec['stage'] = stage
        json.dump(rec, open(os.path.join(wd, 'wave.json'), 'w'))
        with pytest.raises(S.SubmitError, match='a wave per stage'):
            S.load_wave(root, S.RUN_STAGE, wave)
    S.prepare(step, root, {})
    n = len(world['fake'].calls)
    with pytest.raises(S.SubmitError, match='a wave per stage'):
        S.submit(world['lsf'], root, step, 'w4096_30_r002')
    assert not any(c[0] == 'bsub' for c in world['fake'].calls[n:])
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, 'w_r003', step['mem_mb'], step['wall_min'])
    assert S.load_wave(root, S.RUN_STAGE, 'w_r003')['stage'] == 'run_003'
    assert len(S.submit(world['lsf'], root, step, 'w_r003')) == 1


def test_a8_job_dir_carried_from_earlier_stage_refused(world):
    """implementation_008 (Issue B): a job dir carried by lineage (prepared for run_002, self-consistent hashes) is
    never submitted in run_003; prepare cannot overwrite it either."""
    st, root = world['steps'], world['root']
    step = st['m5_tie_hosts']
    S.prepare(step, root, {})
    jd = os.path.join(root, 'run_003', 'jobs', 'm5_tie_hosts')
    sj = os.path.join(jd, 'step.json')
    rec = json.load(open(sj))
    old = json.loads(json.dumps(rec).replace(os.path.join(root, 'run_003'), os.path.join(root, 'run_002')))
    old['stage'] = 'run_002'
    for k in ('stage', 'args'):
        bad = dict(rec, **{k: old[k]})
        os.chmod(sj, 0o644)
        os.remove(sj)
        json.dump(bad, open(sj, 'w'))
        with pytest.raises(S.SubmitError, match=f'prepared step.json {k} '):
            S.verify_prepared(step, root)
    os.remove(sj)
    json.dump(rec, open(sj, 'w'))
    S.verify_prepared(step, root)
    with pytest.raises(Exception):
        S.prepare(step, root, {})                                       # write_new refuses an existing job dir


@pytest.mark.parametrize('sub', ['release', 'migrate', 'monitor'])
def test_a004_cli_stage_choices_exclude_run_001(sub, capsys):
    assert 'run_001' not in S.STAGES
    with pytest.raises(SystemExit) as e:
        S.main([sub, '--stage', 'run_001'])
    assert e.value.code == 2 and "invalid choice: 'run_001'" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- ADDENDUM_004 (impl 009)

REAL_ROOT = S.ROOT
REAL_RUN = os.path.join(REAL_ROOT, S.RUN_STAGE)
RESUB = ('m4_adapter_run_a', 'm5_tie_hosts')


def _write_eval_pins(sdir, packages=None, closure=True):
    os.makedirs(os.path.join(sdir, 'eval_attempt_001'), exist_ok=True)
    json.dump({'status': 'COMPLETE'}, open(os.path.join(sdir, 'eval_attempt_001', 'COMPLETE.json'), 'w'))
    pins = {'packages': packages if packages is not None else {S.RR5_FP32_KEY: {'x': 1}}}
    if closure:
        pins['import_closure'] = {}
    json.dump(pins, open(os.path.join(sdir, 'pins_v2main_001.json'), 'w'))


def _write_run_a(sdir, status='COMPLETE', census='a,b\n1,2\n'):
    d = os.path.join(sdir, 'rr5_additive_001')
    os.makedirs(d, exist_ok=True)
    json.dump({'status': status}, open(os.path.join(d, 'COMPLETE.json'), 'w'))
    if census is not None:
        open(os.path.join(d, 'tie_census.csv'), 'w').write(census)


def _exit_script_job(world, name, jid, rc=1, out=True):
    sdir = os.path.join(world['root'], S.RUN_STAGE)
    if out:
        os.makedirs(os.path.join(sdir, 'lsf'), exist_ok=True)
        open(os.path.join(sdir, 'lsf', f'v2m_{name}.{jid}_0.out'), 'w').write(
            f'Sender: LSF System\nSubject: Job {jid}: <v2m_{name}> Exited\nExited with exit code {rc}.\n')
    world['fake'].jobs[jid]['el'][0] = 'EXIT'


def _reason(world, name, ids, **over):
    d = {'step': name, 'prior_job_ids': [int(x) for x in ids], 'cause': 'synthetic cause', 'addendum': 'ADDENDUM_004',
         'extra': 'kept'}
    d.update(over)
    p = os.path.join(world['root'], 'lead', f'resubmit_reason_{name}.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p):
        os.remove(p)
    open(p, 'w').write(json.dumps(d) if not isinstance(d.get('_raw'), str) else d['_raw'])
    return p


def _resub_world(world, name):
    """Prepared step, its requirement met, one held+released first submission now EXIT rc 1, a valid reason file.
    Returns (step, wave, first job id, reason path)."""
    root, step = world['root'], world['steps'][name]
    sdir = os.path.join(root, S.RUN_STAGE)
    _prepared(world, name)
    if name == 'm4_adapter_run_a':
        _write_eval_pins(sdir)
    else:
        _write_run_a(sdir)
    wave = f'w{step["mem_mb"]}_{step["wall_min"]}_r003'
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, wave, step['mem_mb'], step['wall_min'])
    subs = S.submit(world['lsf'], root, step, wave)
    assert len(subs) == 1
    jid = subs[0]['job_id']
    _exit_script_job(world, name, jid)
    return step, wave, jid, _reason(world, name, [jid])


def _tree(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            out[p] = ('->' + os.readlink(p)).encode() if os.path.islink(p) else open(p, 'rb').read()
        for d in dn:
            out[os.path.join(dp, d) + '/'] = b''
    return out


def _assert_refused(world, name, wave, reason, match, dry_run=False):
    before, n = _tree(world['root']), len(world['fake'].calls)
    with pytest.raises(S.SubmitError, match=match):
        S.resubmit(world['lsf'], world['root'], world['steps'][name], wave, reason, dry_run=dry_run)
    assert _tree(world['root']) == before                         # nothing written
    assert not any(c[0] in ('bsub', 'bresume') for c in world['fake'].calls[n:])


def test_a004_resubmit_constants_and_requires_wiring(world):
    assert S.RUN_STAGE_RESUBMIT_STEPS == ('m4_adapter_run_a', 'm5_tie_hosts')
    assert S.RUN_STAGE_RETRIED_STEPS == ('m3_gates', 'm4_ranks_gate')              # unchanged
    st = world['steps']
    assert st['m4_adapter_run_a']['requires'] == ['eval_pins']
    assert st['m5_tie_hosts']['requires'] == ['rr5_run_a']
    assert st['m5_compare']['requires'] == ['rr5_run_a', 'rr5_run_b']                # ADDENDUM_005 ruling 3
    assert st['m4_adapter_run_b']['requires'] == ['rr5_run_a']                       # ADDENDUM_005 ruling 1
    for n in RESUB:
        assert st[n]['kind'] == 'script' and st[n]['indices'] == [0] and st[n]['stage'] == S.RUN_STAGE
    assert (st['m4_adapter_run_a']['mem_mb'], st['m4_adapter_run_a']['wall_min']) == (4096, 60)
    assert (st['m5_tie_hosts']['mem_mb'], st['m5_tie_hosts']['wall_min']) == (4096, 30)
    assert st['m3_ot_fanout']['requires'] == ['ot_canary'] and st['m3_bob_fanout']['requires'] == ['bob_canary']
    assert all(not v['requires'] for k, v in st.items()
               if k not in ('m4_adapter_run_a', 'm5_tie_hosts', 'm5_compare', 'm3_ot_fanout', 'm3_bob_fanout',
                            'm4_adapter_run_b'))                                     # ADDENDUM_005 ruling 1


def test_a004_fp32_key_mirrors_frozen_adapter_and_tie_census_is_per_cell():
    spec = importlib.util.spec_from_file_location('v2main_rr5_adapter_for_submit_test', S.SCRIPTS['rr5_adapter'])
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    assert S.RR5_FP32_KEY == A.FP32 == 'virchow2_fp32'
    assert 'tie_census.csv' in A.PER_CELL_FILES


def test_a004_check_requires_eval_pins(world):
    root, step = world['root'], world['steps']['m4_adapter_run_a']
    sdir = os.path.join(root, S.RUN_STAGE)
    os.makedirs(sdir)
    with pytest.raises(S.SubmitError, match='eval_attempt_001/COMPLETE.json absent'):
        S.check_requires(step, root)
    _write_eval_pins(sdir)
    S.check_requires(step, root)                                                    # positive
    pp = os.path.join(sdir, 'pins_v2main_001.json')
    os.remove(pp)
    with pytest.raises(S.SubmitError, match='missing or not JSON'):
        S.check_requires(step, root)
    open(pp, 'w').write('{not json')
    with pytest.raises(S.SubmitError, match='missing or not JSON'):
        S.check_requires(step, root)
    for bad in ({'import_closure': {}}, {'packages': [], 'import_closure': {}},
                {'packages': {'virchow2': {}}, 'import_closure': {}}, {'packages': {S.RR5_FP32_KEY: {}}}, []):
        json.dump(bad, open(pp, 'w'))
        with pytest.raises(S.SubmitError, match='lacks packages'):
            S.check_requires(step, root)
    _write_eval_pins(sdir)
    os.remove(os.path.join(sdir, 'eval_attempt_001', 'COMPLETE.json'))
    with pytest.raises(S.SubmitError, match='COMPLETE.json absent'):
        S.check_requires(step, root)


@pytest.mark.parametrize('name', ['m5_tie_hosts', 'm5_compare'])
def test_a004_check_requires_rr5_run_a(world, name):
    root, step = world['root'], world['steps'][name]
    sdir = os.path.join(root, S.RUN_STAGE)
    os.makedirs(sdir)
    if name == 'm5_compare':
        _write_run_b(sdir)          # ADDENDUM_005 ruling 3: compare also requires rr5_run_b (checked after rr5_run_a)
    with pytest.raises(S.SubmitError, match='rr5_run_a: .*missing or not JSON'):
        S.check_requires(step, root)
    _write_run_a(sdir)
    S.check_requires(step, root)                                                    # positive
    cp = os.path.join(sdir, 'rr5_additive_001', 'COMPLETE.json')
    open(cp, 'w').write('{bad')
    with pytest.raises(S.SubmitError, match='missing or not JSON'):
        S.check_requires(step, root)
    for bad in ({'status': 'FAILED'}, {}, ['COMPLETE']):
        json.dump(bad, open(cp, 'w'))
        with pytest.raises(S.SubmitError, match='!= COMPLETE'):
            S.check_requires(step, root)
    _write_run_a(sdir, census='')
    with pytest.raises(S.SubmitError, match='tie_census.csv absent or empty'):
        S.check_requires(step, root)
    os.remove(os.path.join(sdir, 'rr5_additive_001', 'tie_census.csv'))
    with pytest.raises(S.SubmitError, match='tie_census.csv absent or empty'):
        S.check_requires(step, root)


def test_a004_submit_enforces_new_requires(world):
    root = world['root']
    for name in ('m4_adapter_run_a', 'm5_tie_hosts', 'm5_compare'):
        _prepared(world, name)
        step = world['steps'][name]
        wave = f'q_{name}'
        S.probe_wave(world['lsf'], root, S.RUN_STAGE, wave, step['mem_mb'], step['wall_min'])
        n = len(world['fake'].calls)
        with pytest.raises(S.SubmitError, match='requires (eval_pins|rr5_run_a)'):
            S.submit(world['lsf'], root, step, wave)
        assert not any(c[0] == 'bsub' for c in world['fake'].calls[n:])
    _write_eval_pins(os.path.join(root, S.RUN_STAGE))
    _write_run_a(os.path.join(root, S.RUN_STAGE))
    _write_run_b(os.path.join(root, S.RUN_STAGE))                                   # ADDENDUM_005 ruling 3
    for name in ('m4_adapter_run_a', 'm5_tie_hosts', 'm5_compare'):                # positive: requirement met
        assert len(S.submit(world['lsf'], root, world['steps'][name], f'q_{name}')) == 1


@pytest.mark.parametrize('name', RESUB)
def test_a004_resubmit_positive_path(world, name):
    root = world['root']
    step, wave, jid, reason = _resub_world(world, name)
    n = len(world['fake'].calls)
    rec = S.resubmit(world['lsf'], root, step, wave, reason)
    assert set(rec) == {'event', 'step', 'wave', 'job_id', 'queue', 'indices', 'mem_mb', 'wall_min', 'argv', 'held',
                        'jobname', 'prior_job_ids', 'prior_exit_rc', 'reason_file', 'reason_sha256'}
    assert rec['event'] == 'lead_resubmit' and rec['step'] == name and rec['wave'] == wave and rec['held'] is True
    assert rec['indices'] == [0] and rec['jobname'] == f'v2m_{name}' and rec['queue'] == 'standard'
    assert (rec['mem_mb'], rec['wall_min']) == (step['mem_mb'], step['wall_min'])
    assert rec['prior_job_ids'] == [jid] and rec['prior_exit_rc'] == [1] and rec['job_id'] != jid
    assert rec['reason_file'] == os.path.realpath(reason)
    assert rec['reason_sha256'] == hashlib.sha256(open(reason, 'rb').read()).hexdigest()
    c = rec['argv']
    assert c == S.element_bsub_argv(step, os.path.join(root, S.RUN_STAGE), 'standard', [0], step['mem_mb'],
                                    step['wall_min'])
    assert '-H' in c and c[c.index('-M') + 1] == str(step['mem_mb']) and c[c.index('-W') + 1] == str(step['wall_min'])
    assert c[c.index('-J') + 1] == f'v2m_{name}' and c[-1].endswith(f'jobs/{name}/job.sh')
    new = [x for x in world['fake'].calls[n:] if x[0] in ('bsub', 'bresume')]
    assert [x[0] for x in new] == ['bsub', 'bresume'] and new[1] == ['bresume', rec['job_id']]
    jr = S.read_journal(root, S.RUN_STAGE)
    tail = [(r['event'], r.get('job_id')) for r in jr if r.get('job_id') == rec['job_id']]
    assert tail == [('lead_resubmit', rec['job_id']), ('readback_ok', rec['job_id']), ('released', rec['job_id'])]
    lr = [r for r in jr if r.get('event') == 'lead_resubmit']
    assert len(lr) == 1 and {k: v for k, v in lr[0].items() if k != 'time'} == rec
    assert os.path.exists(os.path.join(root, S.RUN_STAGE, 'jobs', name, f'readback_{rec["job_id"]}.json'))
    assert [r['job_id'] for r in S.journaled_elements(root, S.RUN_STAGE, name)[0]] == [jid, rec['job_id']]
    m = len(world['fake'].calls)
    assert S.submit(world['lsf'], root, step, wave) == []                 # lead_resubmit counted: submit refuses
    assert not any(x[0] == 'bsub' for x in world['fake'].calls[m:])
    _exit_script_job(world, name, rec['job_id'])
    _reason(world, name, [jid, rec['job_id']])
    _assert_refused(world, name, wave, reason, 'once only')               # never a second analyst resubmission


def test_a004_resubmit_record_written_before_readback(world):
    """The lead_resubmit record is journaled before readback; a readback failure leaves the job held and named."""
    root = world['root']
    step, wave, jid, reason = _resub_world(world, 'm5_tie_hosts')
    orig = world['fake'].__call__

    def tamper(argv, env):
        rc, out, err = orig(argv, env)
        if argv[0] == 'bsub' and '-H' in argv:
            world['fake'].jobs[re.search(r'<(\d+)>', out).group(1)]['queue'] = 'other'
        return rc, out, err
    world['lsf'].runner = tamper
    n = len(world['fake'].calls)
    with pytest.raises(S.SubmitError, match='left held for the lead'):
        S.resubmit(world['lsf'], root, step, wave, reason)
    assert not any(c[0] == 'bresume' for c in world['fake'].calls[n:])
    ev = [r['event'] for r in S.read_journal(root, S.RUN_STAGE)][-2:]
    assert ev == ['lead_resubmit', 'readback_failed']


@pytest.mark.parametrize('stat', ['DONE', 'RUN', 'PEND', 'PSUSP', 'UNKNOWN', 'EXIT_NO_OUT'])
def test_a004_resubmit_refuses_3a_not_all_exit(world, stat):
    step, wave, jid, reason = _resub_world(world, 'm4_adapter_run_a')
    if stat == 'UNKNOWN':
        del world['fake'].jobs[jid]
    elif stat == 'EXIT_NO_OUT':
        os.remove(os.path.join(world['root'], S.RUN_STAGE, 'lsf', f'v2m_m4_adapter_run_a.{jid}_0.out'))
    else:
        world['fake'].jobs[jid]['el'][0] = stat
    match = {'DONE': 'is DONE, not EXIT', 'UNKNOWN': 'bjobs returned 0 rows',
             'EXIT_NO_OUT': 'LSF output file is missing'}.get(stat, f'is {stat}, not EXIT/DONE')
    _assert_refused(world, 'm4_adapter_run_a', wave, reason, match)


def test_a004_resubmit_refuses_3a_any_prior_not_exit(world):
    """Every journaled submission of the step must be EXIT: a second (synthetic) DONE submission refuses."""
    root = world['root']
    step, wave, jid, reason = _resub_world(world, 'm5_tie_hosts')
    world['fake'].jobs['777'] = {'name': 'v2m_m5_tie_hosts', 'queue': 'standard', 'el': {0: 'DONE'}, 'host': 'h'}
    S.journal(root, S.RUN_STAGE, {'event': 'submit', 'step': 'm5_tie_hosts', 'job_id': '777', 'indices': [0],
                                  'jobname': 'v2m_m5_tie_hosts', 'held': True, 'argv': []})
    reason = _reason(world, 'm5_tie_hosts', [jid, '777'])
    _assert_refused(world, 'm5_tie_hosts', wave, reason, 'journaled job 777 is DONE, not EXIT')


def test_a004_resubmit_refuses_3b_existing_lead_resubmit(world):
    root = world['root']
    step, wave, jid, reason = _resub_world(world, 'm4_adapter_run_a')
    S.journal(root, S.RUN_STAGE, {'event': 'lead_resubmit', 'step': 'm4_adapter_run_a', 'job_id': '999',
                                  'indices': [0], 'held': True})
    _assert_refused(world, 'm4_adapter_run_a', wave, reason, 'once only')


@pytest.mark.parametrize('name,path', [('m4_adapter_run_a', 'rr5_additive_001'),
                                       ('m4_adapter_run_a', 'rr5_additive_001.full_spec.json'),
                                       ('m4_adapter_run_a', 'rr5_additive_001_recompute_bindings'),
                                       ('m5_tie_hosts', 'tie_census_host_diversity.json')])
def test_a004_resubmit_refuses_3c_outputs_present(world, name, path):
    step, wave, jid, reason = _resub_world(world, name)
    p = os.path.join(world['root'], S.RUN_STAGE, path)
    if path.endswith('.json'):
        open(p, 'w').write('{}')
    elif not os.path.exists(p):
        os.makedirs(p)
    _assert_refused(world, name, wave, reason, 'output path')


def test_a004_resubmit_3c_dangling_symlink_counts_as_present(world):
    step, wave, jid, reason = _resub_world(world, 'm4_adapter_run_a')
    os.symlink(os.path.join(world['root'], 'nowhere'), os.path.join(world['root'], S.RUN_STAGE,
                                                                    'rr5_additive_001.full_spec.json'))
    _assert_refused(world, 'm4_adapter_run_a', wave, reason, 'output path')


@pytest.mark.parametrize('name', RESUB)
def test_a004_resubmit_refuses_3d_requires(world, name):
    step, wave, jid, reason = _resub_world(world, name)
    sdir = os.path.join(world['root'], S.RUN_STAGE)
    if name == 'm4_adapter_run_a':
        os.remove(os.path.join(sdir, 'eval_attempt_001', 'COMPLETE.json'))
        match = 'requires eval_pins'
    else:
        os.remove(os.path.join(sdir, 'rr5_additive_001', 'tie_census.csv'))
        match = 'requires rr5_run_a'
    _assert_refused(world, name, wave, reason, match)


@pytest.mark.parametrize('mark', ['FAILED', 'COMPLETE.json', 'PARTIAL.json'])
def test_a004_resubmit_refuses_3e_stage_markers(world, mark):
    step, wave, jid, reason = _resub_world(world, 'm5_tie_hosts')
    open(os.path.join(world['root'], S.RUN_STAGE, mark), 'w').write('x')
    _assert_refused(world, 'm5_tie_hosts', wave, reason, f"carries \\['{re.escape(mark)}'\\]")


@pytest.mark.parametrize('case', ['outside', 'symlink_out', 'missing', 'dir', 'not_json', 'not_object', 'step',
                                  'ids_mismatch', 'ids_superset', 'ids_empty', 'ids_not_list', 'cause_empty',
                                  'cause_missing', 'addendum'])
def test_a004_resubmit_refuses_3f_reason_file(world, case):
    root = world['root']
    name = 'm4_adapter_run_a'
    step, wave, jid, reason = _resub_world(world, name)
    lead = os.path.join(root, 'lead')
    match = 'reason file'
    if case == 'outside':
        reason2 = os.path.join(root, 'elsewhere.json')
        open(reason2, 'w').write(open(reason).read())
        reason, match = reason2, 'is not under'
    elif case == 'symlink_out':
        out = os.path.join(root, 'elsewhere.json')
        open(out, 'w').write(open(reason).read())
        reason = os.path.join(lead, 'link.json')
        os.symlink(out, reason)
        match = 'is not under'
    elif case == 'missing':
        reason, match = os.path.join(lead, 'absent.json'), 'not a regular file'
    elif case == 'dir':
        reason = os.path.join(lead, 'd')
        os.makedirs(reason)
        match = 'not a regular file'
    elif case == 'not_json':
        reason, match = _reason(world, name, [jid], _raw='{x'), 'is not JSON'
    elif case == 'not_object':
        reason, match = _reason(world, name, [jid], _raw='[1]'), 'not a JSON object'
    elif case == 'step':
        reason, match = _reason(world, name, [jid], step='m5_tie_hosts'), 'step'
    elif case == 'ids_mismatch':
        reason, match = _reason(world, name, ['12345']), 'prior_job_ids .* != journaled'
    elif case == 'ids_superset':
        reason, match = _reason(world, name, [jid, '12345']), 'prior_job_ids .* != journaled'
    elif case == 'ids_empty':
        reason, match = _reason(world, name, []), 'non-empty list'
    elif case == 'ids_not_list':
        reason, match = _reason(world, name, [jid], prior_job_ids=int(jid)), 'non-empty list'
    elif case == 'cause_empty':
        reason, match = _reason(world, name, [jid], cause='  '), 'cause'
    elif case == 'cause_missing':
        reason = _reason(world, name, [jid])
        d = json.load(open(reason))
        del d['cause']
        reason, match = _reason(world, name, [], _raw=json.dumps(d)), 'cause'
    elif case == 'addendum':
        reason, match = _reason(world, name, [jid], addendum='ADDENDUM_003'), 'addendum'
    _assert_refused(world, name, wave, reason, match)
    _assert_refused(world, name, wave, reason, match, dry_run=True)           # file-only: also refused in dry-run


def test_a004_resubmit_reason_ids_as_strings_accepted(world):
    """prior_job_ids are compared as job-id strings (the real analyst files carry ints; the journal carries strings)."""
    step, wave, jid, reason = _resub_world(world, 'm5_tie_hosts')
    p = _reason(world, 'm5_tie_hosts', [jid], prior_job_ids=[str(jid)])
    assert S.resubmit(world['lsf'], world['root'], step, wave, p)['prior_job_ids'] == [jid]


def test_a004_resubmit_refuses_wrong_step_wave_and_never_submitted(world):
    root = world['root']
    step, wave, jid, reason = _resub_world(world, 'm4_adapter_run_a')
    for other in ('m4_ranks_gate', 'm5_compare', 'm3_gates'):
        with pytest.raises(S.SubmitError, match='resubmit applies only'):
            S.resubmit(world['lsf'], root, world['steps'][other], wave, reason)
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, 'w_other_shape', 4096, 30)
    _assert_refused(world, 'm4_adapter_run_a', 'w_other_shape', reason, 'M0.4.2')
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, 'w_retry_shape', 4096, 60, kind='retry')
    _assert_refused(world, 'm4_adapter_run_a', 'w_retry_shape', reason, 'M0.4.2')
    _assert_refused(world, 'm4_adapter_run_a', 'w_absent', reason, 'wave w_absent unreadable')
    os.remove(os.path.join(root, S.RUN_STAGE, 'jobs', 'm4_adapter_run_a', 'step.json'))
    _assert_refused(world, 'm4_adapter_run_a', wave, reason, 'prepared job files.*unreadable')


def test_a004_resubmit_refuses_never_submitted_step(world):
    root = world['root']
    _prepared(world, 'm5_tie_hosts')
    _write_run_a(os.path.join(root, S.RUN_STAGE))
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, 'w', 4096, 30)
    reason = _reason(world, 'm5_tie_hosts', ['1'])
    _assert_refused(world, 'm5_tie_hosts', 'w', reason, 'no journaled submission')


def test_a004_resubmit_reuses_prepared_files_only(world):
    step, wave, jid, reason = _resub_world(world, 'm5_tie_hosts')
    js = os.path.join(world['root'], S.RUN_STAGE, 'jobs', 'm5_tie_hosts', 'job.sh')
    os.chmod(js, 0o644)
    open(js, 'a').write('# tampered\n')
    _assert_refused(world, 'm5_tie_hosts', wave, reason, 'changed since prepare')


@pytest.mark.parametrize('name', RESUB)
def test_a004_resubmit_dry_run_writes_nothing(world, name, capsys):
    step, wave, jid, reason = _resub_world(world, name)
    capsys.readouterr()
    before, n = _tree(world['root']), len(world['fake'].calls)
    assert S.resubmit(world['lsf'], world['root'], step, wave, reason, dry_run=True) is None
    assert _tree(world['root']) == before and len(world['fake'].calls) == n        # no LSF call, no write
    out = capsys.readouterr().out
    assert out.startswith('DRY resubmit') and f'-J v2m_{name}' in out and ' -H ' in out


def test_a004_resubmit_cli(world, capsys):
    step, wave, jid, reason = _resub_world(world, 'm4_adapter_run_a')
    before = _tree(world['root'])
    capsys.readouterr()
    assert S.main(['--root', world['root'], '--dry-run', 'resubmit', '--step', 'm4_adapter_run_a', '--wave', wave,
                   '--reason-file', reason]) == 0
    assert 'DRY resubmit' in capsys.readouterr().out and _tree(world['root']) == before
    assert S.main(['--root', world['root'], '--dry-run', 'resubmit', '--step', 'm4_adapter_run_a', '--wave', wave,
                   '--reason-file', '/nonexistent']) == 1
    assert 'STOP: resubmit m4_adapter_run_a: reason file /nonexistent is not under' in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        S.main(['--root', world['root'], 'resubmit', '--step', 'm4_ranks_gate', '--wave', wave,
                '--reason-file', reason])
    assert e.value.code == 2


@pytest.mark.parametrize('name', RESUB)
def test_a004_retry_refuses_resubmit_steps_first(world, name):
    step, wave, jid, reason = _resub_world(world, name)
    S.probe_wave(world['lsf'], world['root'], S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    before, n = _tree(world['root']), len(world['fake'].calls)
    with pytest.raises(S.SubmitError, match=f'use `resubmit --step {name}'):
        S.retry(world['lsf'], world['root'], world['steps'][name], 0, 'r01')
    assert _tree(world['root']) == before and len(world['fake'].calls) == n
    assert not os.path.exists(os.path.join(world['root'], S.RUN_STAGE, 'failed_elements.jsonl'))


def test_a004_journaled_elements_counts_lead_resubmit(world):
    root = world['root']
    os.makedirs(os.path.join(root, S.RUN_STAGE))
    S.journal(root, S.RUN_STAGE, {'event': 'lead_resubmit', 'step': 'm5_tie_hosts', 'job_id': '5', 'indices': [0]})
    S.journal(root, S.RUN_STAGE, {'event': 'readback_ok', 'step': 'm5_tie_hosts', 'job_id': '5'})
    assert [r['job_id'] for r in S.journaled_elements(root, S.RUN_STAGE, 'm5_tie_hosts')[0]] == ['5']


# ---- real-shape, read-only checks against the actual run_003 package (skipped if unavailable)

def _real(p):
    if not os.path.exists(p):
        pytest.skip(f'real file unavailable: {p}')
    return p


def test_a004_real_journal_copy_has_one_submit_per_resubmit_step(tmp_path):
    src = _real(os.path.join(REAL_RUN, 'journal.jsonl'))
    root = str(tmp_path / 'pkg')
    os.makedirs(os.path.join(root, S.RUN_STAGE))
    open(os.path.join(root, S.RUN_STAGE, 'journal.jsonl'), 'wb').write(open(src, 'rb').read())
    jr = S.read_journal(root, S.RUN_STAGE)
    assert not any(r.get('event') == 'lead_resubmit' for r in jr)
    for name, jid in (('m5_tie_hosts', '332617057'), ('m4_adapter_run_a', '332620328')):
        subs = [r for r in jr if r.get('event') == 'submit' and r.get('step') == name]
        assert [str(r['job_id']) for r in subs] == [jid]
        assert [str(r['job_id']) for r in S.journaled_elements(root, S.RUN_STAGE, name)[0]] == [jid]


def test_a004_real_pins_satisfy_eval_pins(tmp_path):
    pins = _real(os.path.join(REAL_RUN, 'pins_v2main_001.json'))
    comp = _real(os.path.join(REAL_RUN, 'eval_attempt_001', 'COMPLETE.json'))
    root = str(tmp_path / 'pkg')
    sdir = os.path.join(root, S.RUN_STAGE)
    os.makedirs(os.path.join(sdir, 'eval_attempt_001'))
    open(os.path.join(sdir, 'pins_v2main_001.json'), 'wb').write(open(pins, 'rb').read())
    open(os.path.join(sdir, 'eval_attempt_001', 'COMPLETE.json'), 'wb').write(open(comp, 'rb').read())
    S.check_requires(S.build_steps(root)['m4_adapter_run_a'], root)


@pytest.mark.parametrize('name,jid', [('m4_adapter_run_a', '332620328'), ('m5_tie_hosts', '332617057')])
def test_a004_real_reason_files_and_out_files(tmp_path, name, jid):
    """The analyst's real reason files pass 3f against a copy of the real journal; the real out file parses rc 1."""
    src = _real(os.path.join(REAL_ROOT, 'lead', f'resubmit_reason_{name}.json'))
    jsrc = _real(os.path.join(REAL_RUN, 'journal.jsonl'))
    out = _real(os.path.join(REAL_RUN, 'lsf', f'v2m_{name}.{jid}_0.out'))
    root = str(tmp_path / 'pkg')
    os.makedirs(os.path.join(root, 'lead'))
    os.makedirs(os.path.join(root, S.RUN_STAGE))
    dst = os.path.join(root, 'lead', os.path.basename(src))
    open(dst, 'wb').write(open(src, 'rb').read())
    open(os.path.join(root, S.RUN_STAGE, 'journal.jsonl'), 'wb').write(open(jsrc, 'rb').read())
    rp, sha, d, ids = S._load_reason(root, name, dst)
    assert sha == hashlib.sha256(open(src, 'rb').read()).hexdigest() and ids == {jid}
    assert ids == {str(r['job_id']) for r in S.journaled_elements(root, S.RUN_STAGE, name)[0]}
    assert S._fid().parse_lsf_out(open(out, errors='replace').read())['rc'] == 1


@pytest.mark.parametrize('name,wave', [('m4_adapter_run_a', 'w4096_60_r003'), ('m5_tie_hosts', 'w4096_30_r003')])
def test_a004_real_package_dry_run_read_only(name, wave, capsys):
    """Dry-run resubmit against the real package: reads only (prepared jobs, waves, pins, journal, reason files);
    no LSF call (a runner that fails on any call) and the run_003 tree is unchanged. m5_tie_hosts is expected to stop
    at 3d until run-a has produced rr5_additive_001 (ADDENDUM_004 ruling 4 ordering)."""
    reason = _real(os.path.join(REAL_ROOT, 'lead', f'resubmit_reason_{name}.json'))
    _real(os.path.join(REAL_RUN, 'queue_probe', wave, 'wave.json'))

    def no_lsf(argv, env):
        raise AssertionError(f'LSF called in dry-run: {argv}')
    lsf = S.Lsf(runner=no_lsf)
    listing = sorted(os.listdir(REAL_RUN))
    jl = open(os.path.join(REAL_RUN, 'journal.jsonl'), 'rb').read()
    step = S.build_steps(REAL_ROOT)[name]
    rr5 = os.path.join(REAL_RUN, 'rr5_additive_001', 'COMPLETE.json')
    if name == 'm5_tie_hosts' and not os.path.exists(rr5):
        with pytest.raises(S.SubmitError, match='requires rr5_run_a'):
            S.resubmit(lsf, REAL_ROOT, step, wave, reason, dry_run=True)
    else:
        try:
            S.resubmit(lsf, REAL_ROOT, step, wave, reason, dry_run=True)
        except S.SubmitError as e:
            pytest.skip(f'real package no longer in the pre-resubmission state: {e}')
        assert capsys.readouterr().out.startswith('DRY resubmit')
    assert sorted(os.listdir(REAL_RUN)) == listing and open(os.path.join(REAL_RUN, 'journal.jsonl'), 'rb').read() == jl


# ---------------------------------------------------------------------------------------------- ADDENDUM_005 (impl 010)

RUN_B_SIGNOFF_SHA = 'c87762d8cf975aa159debe839b6547ce0f436ede6c5356f77d9a7785b8974d4f'
RUN_B_WAVE = 'w8192_60_r003'


def _write_run_b(sdir, report=None, raw=None, failed=False):
    """rr5_replacement_001 with a faithfulness report (default: a passing one); optionally a FAILED.json."""
    d = os.path.join(sdir, 'rr5_replacement_001')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'faithfulness_report.json')
    if os.path.lexists(p):
        os.remove(p)
    open(p, 'w').write(raw if raw is not None else json.dumps(report if report is not None
                                                             else {'pass': True, 'failures': []}))
    if failed:
        json.dump({'status': 'FAILED'}, open(os.path.join(d, 'FAILED.json'), 'w'))
    return d


def _load_frozen_adapter(tag):
    spec = importlib.util.spec_from_file_location(f'v2main_rr5_adapter_for_submit_test_{tag}', S.SCRIPTS['rr5_adapter'])
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    return A


def test_a005_run_b_step_exact_attributes(world):
    st, root = world['steps'], world['root']
    rd = os.path.join(root, S.RUN_STAGE)
    s = st['m4_adapter_run_b']
    assert s == {'name': 'm4_adapter_run_b', 'stage': S.RUN_STAGE, 'kind': 'script', 'mem_mb': 8192, 'wall_min': 60,
                 'jobname': 'v2m_m4_adapter_run_b', 'indices': [0], 'one_element_arrays': False,
                 'args': [S.SCRIPTS['rr5_adapter'], 'run-b', '--run-dir', rd, '--signoff',
                          os.path.join(root, 'RUN_B_SIGNOFF.md')],
                 'module': None, 'params': None, 'i1': list(S.RR5_I1), 'own_scripts': ['rr5_adapter'], 'pre': [],
                 'mem_from': None, 'requires': ['rr5_run_a'], 'jobfile': None}
    assert S.RR5_I1 == (24, 26)
    assert '--feasibility' not in s['args'] and '--run-b-pending' not in s['args']     # adapter default feasibility
    assert not S.PROTECTED_NAME_RE.search(s['jobname'])
    assert 'm4_adapter_run_b' not in S.RUN_STAGE_RESUBMIT_STEPS + S.RUN_STAGE_RETRIED_STEPS
    assert S.RUN_STAGE_RESUBMIT_STEPS == ('m4_adapter_run_a', 'm5_tie_hosts')
    assert S.RUN_STAGE_RETRIED_STEPS == ('m3_gates', 'm4_ranks_gate')
    assert 'm4_adapter_run_b' not in S.RESUBMIT_OUTPUTS
    c = st['m5_compare']
    assert c['requires'] == ['rr5_run_a', 'rr5_run_b']
    assert c['args'] == [S.SCRIPTS['compare'], '--run-dir', rd]                       # unchanged, no --run-b-pending
    assert (c['kind'], c['mem_mb'], c['wall_min'], c['i1'], c['own_scripts'], c['jobname']) == \
        ('script', 8192, 60, [24, 26], ['compare', 'rr5_adapter'], 'v2m_m5_compare')
    a = st['m4_adapter_run_a']
    assert a['args'] == [S.SCRIPTS['rr5_adapter'], 'run-a', '--run-dir', rd] and a['requires'] == ['eval_pins']
    assert st['m5_tie_hosts']['requires'] == ['rr5_run_a']


def test_a005_run_b_argv_parses_under_frozen_adapter(world, monkeypatch):
    """The step argv is accepted by the frozen adapter's own `run-b` parser (cmd_run_b replaced: nothing runs)."""
    A = _load_frozen_adapter('a005')
    seen = []
    monkeypatch.setattr(A, 'cmd_run_b', lambda args: seen.append(args) or 0)
    s = world['steps']['m4_adapter_run_b']
    assert s['args'][0] == S.SCRIPTS['rr5_adapter']
    assert A.main(s['args'][1:]) == 0
    (a,) = seen
    assert a.cmd == 'run-b' and a.run_dir == os.path.join(world['root'], S.RUN_STAGE)
    assert a.signoff == os.path.join(world['root'], 'RUN_B_SIGNOFF.md')
    assert A.ROOT == S.ROOT and a.feasibility == os.path.join(S.ROOT, 'implementation_001', 'RUN_B_FEASIBILITY.md')
    assert a.n_reps == 5000 and a.run001 == A.RUN001 and (a.rr5, a.rr5_sha) == (A.RR5_PATH, A.RR5_SHA)


def test_a005_run_b_prepared_job_executes_adapter_run_b(world):
    import shlex
    _prepared(world, 'm4_adapter_run_b')
    s = world['steps']['m4_adapter_run_b']
    jdir = os.path.join(world['root'], S.RUN_STAGE, 'jobs', 'm4_adapter_run_b')
    job = open(os.path.join(jdir, 'job.sh')).read()
    assert shlex.split(job.rstrip('\n').splitlines()[-1]) == ['exec', '$PROSPER_PYTHON'] + s['args']
    assert 'host_sidecar' in job and '"v2m_m4_adapter_run_b.${LSB_JOBID:-none}_${LSB_JOBINDEX:-0}.json"' in job
    assert 'import_closure' not in job                         # script kind: no per-job import-closure sidecar
    rec = json.load(open(os.path.join(jdir, 'step.json')))
    assert rec['args'] == s['args'] and (rec['mem_mb'], rec['wall_min'], rec['kind']) == (8192, 60, 'script')
    assert [g['path'] for g in rec['guard']][-1] == S.SCRIPTS['rr5_adapter'] and len(rec['guard']) == 3
    S.verify_prepared(s, world['root'])


def test_a005_check_requires_rr5_run_b(world):
    root, step = world['root'], world['steps']['m5_compare']
    sdir = os.path.join(root, S.RUN_STAGE)
    rb = os.path.join(sdir, 'rr5_replacement_001')
    fr = os.path.join(rb, 'faithfulness_report.json')
    os.makedirs(sdir)
    _write_run_a(sdir)
    with pytest.raises(S.SubmitError, match='m5_compare: requires rr5_run_b: .*faithfulness_report.json missing or '
                                            'not JSON'):
        S.check_requires(step, root)                                                # rr5_replacement_001 absent
    os.makedirs(rb)
    with pytest.raises(S.SubmitError, match='requires rr5_run_b: .*missing or not JSON'):
        S.check_requires(step, root)                                                # directory without report
    os.makedirs(fr)
    with pytest.raises(S.SubmitError, match='requires rr5_run_b: .*missing or not JSON'):
        S.check_requires(step, root)                                                # report path is a directory
    os.rmdir(fr)
    for raw in ('', '{bad', 'pass: true', '{"pass": true'):
        _write_run_b(sdir, raw=raw)
        with pytest.raises(S.SubmitError, match='requires rr5_run_b: .*missing or not JSON'):
            S.check_requires(step, root)
    for rep in ([], [{'pass': True}], 'true', True, None, {}, {'failures': []}, {'pass': False}, {'pass': None},
                {'pass': 'true'}, {'pass': 1}, {'pass': 1.0}, {'pass': [True]}):
        _write_run_b(sdir, raw=json.dumps(rep))
        with pytest.raises(S.SubmitError, match='requires rr5_run_b: .* is not true'):
            S.check_requires(step, root)
    _write_run_b(sdir)
    S.check_requires(step, root)                                                    # positive
    _write_run_b(sdir, failed=True)
    with pytest.raises(S.SubmitError, match='requires rr5_run_b: .*rr5_replacement_001/FAILED.json present'):
        S.check_requires(step, root)
    fp = os.path.join(rb, 'FAILED.json')
    os.remove(fp)
    os.symlink(os.path.join(rb, 'no_such_target'), fp)                              # dangling: lexists
    with pytest.raises(S.SubmitError, match='FAILED.json present'):
        S.check_requires(step, root)
    os.remove(fp)
    os.makedirs(fp)
    with pytest.raises(S.SubmitError, match='FAILED.json present'):
        S.check_requires(step, root)
    os.rmdir(fp)
    S.check_requires(step, root)                                                    # positive again
    os.remove(os.path.join(sdir, 'rr5_additive_001', 'COMPLETE.json'))
    with pytest.raises(S.SubmitError, match='m5_compare: requires rr5_run_a'):
        S.check_requires(step, root)                                                # rr5_run_a still required first


def test_a005_check_requires_run_b_step_needs_only_run_a(world):
    root, step = world['root'], world['steps']['m4_adapter_run_b']
    sdir = os.path.join(root, S.RUN_STAGE)
    os.makedirs(sdir)
    with pytest.raises(S.SubmitError, match='m4_adapter_run_b: requires rr5_run_a: .*missing or not JSON'):
        S.check_requires(step, root)
    _write_run_a(sdir, status='FAILED')
    with pytest.raises(S.SubmitError, match='m4_adapter_run_b: requires rr5_run_a: .*!= COMPLETE'):
        S.check_requires(step, root)
    _write_run_a(sdir, census='')
    with pytest.raises(S.SubmitError, match='m4_adapter_run_b: requires rr5_run_a: .*tie_census.csv absent or empty'):
        S.check_requires(step, root)
    _write_run_a(sdir)
    S.check_requires(step, root)                                                    # Run B itself absent: fine
    assert not os.path.lexists(os.path.join(sdir, 'rr5_replacement_001'))


def _assert_submit_refused(world, step, wave, match):
    before, n = _tree(world['root']), len(world['fake'].calls)
    with pytest.raises(S.SubmitError, match=match):
        S.submit(world['lsf'], world['root'], step, wave)
    assert _tree(world['root']) == before                                          # nothing written
    assert not any(c[0] in ('bsub', 'bresume') for c in world['fake'].calls[n:])


def _assert_one_submission(world, step, wave):
    n = len(world['fake'].calls)
    subs = S.submit(world['lsf'], world['root'], step, wave)
    assert len(subs) == 1 and subs[0]['event'] == 'submit' and subs[0]['step'] == step['name']
    new = [c for c in world['fake'].calls[n:] if c[0] in ('bsub', 'bresume')]
    assert [c[0] for c in new] == ['bsub', 'bresume'] and new[1] == ['bresume', subs[0]['job_id']]
    c = new[0]
    assert '-H' in c and c[c.index('-J') + 1] == step['jobname']
    assert (c[c.index('-M') + 1], c[c.index('-W') + 1]) == ('8192', '60')
    assert c[-1] == os.path.join(world['root'], S.RUN_STAGE, 'jobs', step['name'], 'job.sh')
    jr = [r for r in S.read_journal(world['root'], S.RUN_STAGE) if r.get('step') == step['name']]
    assert [r['event'] for r in jr if r['event'] == 'submit'] == ['submit']
    n = len(world['fake'].calls)
    assert S.submit(world['lsf'], world['root'], step, wave) == []                 # never duplicated
    assert not any(c[0] == 'bsub' for c in world['fake'].calls[n:])
    return subs[0]


def test_a005_submit_m5_compare_refuses_until_run_b(world):
    import shutil
    root = world['root']
    sdir = os.path.join(root, S.RUN_STAGE)
    step = world['steps']['m5_compare']
    _prepared(world, 'm5_compare')
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, RUN_B_WAVE, 8192, 60)
    _assert_submit_refused(world, step, RUN_B_WAVE, 'm5_compare: requires rr5_run_a')
    _write_run_b(sdir)
    _assert_submit_refused(world, step, RUN_B_WAVE, 'm5_compare: requires rr5_run_a')  # Run B alone is not enough
    shutil.rmtree(os.path.join(sdir, 'rr5_replacement_001'))
    _write_run_a(sdir)
    _assert_submit_refused(world, step, RUN_B_WAVE, 'm5_compare: requires rr5_run_b: .*missing or not JSON')
    _write_run_b(sdir, report={'pass': False, 'failures': ['x']})
    _assert_submit_refused(world, step, RUN_B_WAVE, 'm5_compare: requires rr5_run_b: .* is not true')
    _write_run_b(sdir, failed=True)
    _assert_submit_refused(world, step, RUN_B_WAVE, 'm5_compare: requires rr5_run_b: .*FAILED.json present')
    os.remove(os.path.join(sdir, 'rr5_replacement_001', 'FAILED.json'))
    rec = _assert_one_submission(world, step, RUN_B_WAVE)                           # both requirements hold
    assert '--run-b-pending' not in open(os.path.join(sdir, 'jobs', 'm5_compare', 'job.sh')).read()
    assert rec['wave'] == RUN_B_WAVE


def test_a005_submit_m4_adapter_run_b_after_run_a(world):
    root = world['root']
    sdir = os.path.join(root, S.RUN_STAGE)
    step = world['steps']['m4_adapter_run_b']
    _prepared(world, 'm4_adapter_run_b')
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, RUN_B_WAVE, 8192, 60)
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, 'w4096_60_r003', 4096, 60)
    _assert_submit_refused(world, step, RUN_B_WAVE, 'm4_adapter_run_b: requires rr5_run_a: .*missing or not JSON')
    _write_run_a(sdir, status='FAILED')
    _assert_submit_refused(world, step, RUN_B_WAVE, 'm4_adapter_run_b: requires rr5_run_a: .*!= COMPLETE')
    _write_run_a(sdir)
    _assert_submit_refused(world, step, 'w4096_60_r003', r'M0\.4\.2')               # wrong wave shape
    assert not os.path.lexists(os.path.join(sdir, 'rr5_replacement_001'))
    rec = _assert_one_submission(world, step, RUN_B_WAVE)                           # rr5_run_a holds
    assert rec['queue'] == 'standard' and rec['indices'] == [0] and rec['jobname'] == 'v2m_m4_adapter_run_b'
    m5 = world['steps']['m5_compare']                                               # Run B job submitted, not run
    with pytest.raises(S.SubmitError, match='m5_compare: requires rr5_run_b'):
        S.check_requires(m5, root)


def test_a005_retry_and_resubmit_refuse_run_b(world, capsys):
    root = world['root']
    sdir = os.path.join(root, S.RUN_STAGE)
    step = world['steps']['m4_adapter_run_b']
    _prepared(world, 'm4_adapter_run_b')
    _write_run_a(sdir)
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, RUN_B_WAVE, 8192, 60)
    S.probe_wave(world['lsf'], root, S.RUN_STAGE, 'r01', S.RETRY_MEM, S.RETRY_WALL, kind='retry')
    reason = _reason(world, 'm4_adapter_run_b', ['1'])

    def refused(fn, match):
        before, n = _tree(root), len(world['fake'].calls)
        with pytest.raises(S.SubmitError, match=match):
            fn()
        assert _tree(root) == before and not any(c[0] in ('bsub', 'bresume') for c in world['fake'].calls[n:])
        assert not os.path.exists(os.path.join(sdir, 'failed_elements.jsonl'))

    for dry in (False, True):
        refused(lambda: S.resubmit(world['lsf'], root, step, RUN_B_WAVE, reason, dry_run=dry),
                'resubmit applies only to .*not m4_adapter_run_b; nothing written')
    with pytest.raises(SystemExit) as e:
        S.main(['--root', root, 'resubmit', '--step', 'm4_adapter_run_b', '--wave', RUN_B_WAVE,
                '--reason-file', reason])
    assert e.value.code == 2 and "invalid choice: 'm4_adapter_run_b'" in capsys.readouterr().err
    refused(lambda: S.retry(world['lsf'], root, step, 0, 'r01'), r'm4_adapter_run_b\[0\] was never submitted')
    jid = S.submit(world['lsf'], root, step, RUN_B_WAVE)[0]['job_id']
    for stat in ('PEND', 'RUN', 'PSUSP'):
        world['fake'].jobs[jid]['el'][0] = stat
        refused(lambda: S.retry(world['lsf'], root, step, 0, 'r01'), f'is {stat}, not EXIT/DONE; no retry')
    world['fake'].jobs[jid]['el'][0] = 'DONE'
    refused(lambda: S.retry(world['lsf'], root, step, 0, 'r01'), 'completed successfully; no retry')
    world['fake'].jobs[jid]['el'][0] = 'EXIT'                                       # EXIT without the LSF .out file
    refused(lambda: S.retry(world['lsf'], root, step, 0, 'r01'), 'LSF output file is missing')
    for dry in (False, True):
        refused(lambda: S.resubmit(world['lsf'], root, step, RUN_B_WAVE, reason, dry_run=dry),
                'resubmit applies only to')


# ---- ADDENDUM_005 real-shape, read-only checks against the actual package (skipped if unavailable)

def test_a005_real_run_b_argv_names_real_signoff():
    s = S.build_steps()['m4_adapter_run_b']
    sig = s['args'][s['args'].index('--signoff') + 1]
    assert sig == os.path.join(REAL_ROOT, 'RUN_B_SIGNOFF.md') and sig.startswith('/research_jude/')
    assert s['args'][s['args'].index('--run-dir') + 1] == REAL_RUN
    _real(sig)
    assert os.path.isfile(sig) and not os.path.islink(sig)
    assert hashlib.sha256(open(sig, 'rb').read()).hexdigest() == RUN_B_SIGNOFF_SHA


def test_a005_real_signoff_passes_frozen_adapter_signoff_check():
    """The frozen adapter's A2 sign-off check (a read of two small documents; no Run B) accepts the real sign-off
    with the default feasibility file the step relies on."""
    sig = _real(os.path.join(REAL_ROOT, 'RUN_B_SIGNOFF.md'))
    feas = _real(os.path.join(REAL_ROOT, 'implementation_001', 'RUN_B_FEASIBILITY.md'))
    A = _load_frozen_adapter('a005_real')
    b = A.check_signoff(sig, feas)
    assert b['sha256'] == RUN_B_SIGNOFF_SHA and b['path'] == sig and b['feasibility_path'] == feas


def test_a005_real_run_a_satisfies_rr5_run_a_and_run_b_absent():
    _real(os.path.join(REAL_RUN, 'rr5_additive_001', 'COMPLETE.json'))
    steps = S.build_steps(REAL_ROOT)
    S.check_requires(steps['m4_adapter_run_b'], REAL_ROOT)    # read-only: COMPLETE.json + tie_census.csv size
    rb = os.path.join(REAL_RUN, 'rr5_replacement_001')
    if os.path.lexists(rb):
        pytest.skip(f'real package past the pre-Run-B state: {rb} exists')
    with pytest.raises(S.SubmitError, match='m5_compare: requires rr5_run_b: .*missing or not JSON'):
        S.check_requires(steps['m5_compare'], REAL_ROOT)


def test_a005_real_dry_run_prepare_and_submit_write_nothing(capsys):
    """`prepare`/`submit` --dry-run of m4_adapter_run_b against the real package: reads only, no LSF call."""
    _real(os.path.join(REAL_RUN, 'queue_probe', RUN_B_WAVE, 'wave.json'))

    def no_lsf(argv, env):
        raise AssertionError(f'LSF called in dry-run: {argv}')
    listing = sorted(os.listdir(REAL_RUN))
    jobs = sorted(os.listdir(os.path.join(REAL_RUN, 'jobs')))
    jl = open(os.path.join(REAL_RUN, 'journal.jsonl'), 'rb').read()
    step = S.build_steps(REAL_ROOT)['m4_adapter_run_b']
    rec = S.prepare(step, REAL_ROOT, S.load_i1(), dry_run=True)
    assert rec['args'] == step['args'] and set(rec['files']) == {'launch.sha256', 'job.sh'}
    capsys.readouterr()
    S.submit(S.Lsf(runner=no_lsf), REAL_ROOT, step, RUN_B_WAVE, dry_run=True)
    out = capsys.readouterr().out
    assert out.startswith('DRY: bsub -H -q standard -J v2m_m4_adapter_run_b ') or 'already journaled' in out
    assert sorted(os.listdir(REAL_RUN)) == listing and sorted(os.listdir(os.path.join(REAL_RUN, 'jobs'))) == jobs
    assert open(os.path.join(REAL_RUN, 'journal.jsonl'), 'rb').read() == jl
