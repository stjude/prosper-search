"""Synthetic tests for scripts/r2_1_virchow2_fp32_main_gates.py (tiny N, no LSF, no real banks/matrices)."""
import importlib.util
import json
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('v2main_gates_under_test',
                                               os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_gates.py'))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
F = G.F

N = 45                     # 45 rows -> 3 BoB batches (20, 20, 5)
IDS = [str(1000 + i) for i in range(N)]
PATIENT = {s: f'P{i // 3}' for i, s in enumerate(IDS)}       # 3 slides per patient


def pk(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as fh:
        pickle.dump(obj, fh)


def lsf_out(rd, jobname, jid, idx, rc=0, mem=1000, host='nodeA', stdout='ok\n'):
    os.makedirs(os.path.join(rd, 'lsf'), exist_ok=True)
    tail = 'Successfully completed.' if rc == 0 else f'Exited with exit code {rc}.'
    open(os.path.join(rd, 'lsf', f'{jobname}.{jid}_{idx}.out'), 'w').write(
        f'{stdout}Sender: LSF System <x>\nSubject: Job {jid}[{idx}]: <{jobname}> Done\n'
        f'Job was executed on host(s) <{host}>, in queue <standard>, as user\n{tail}\n'
        f'    Max Memory :   {mem} MB\n    Run time :   10 sec.\n')


def lsf_err(rd, jobname, jid, idx, text=''):
    open(os.path.join(rd, 'lsf', f'{jobname}.{jid}_{idx}.err'), 'w').write(text)


def sidecar(rd, jobname, jid, idx, host='nodeA', cpu='AMD EPYC 7713'):
    pk_dir = os.path.join(rd, 'host_sidecar')
    os.makedirs(pk_dir, exist_ok=True)
    json.dump({'hostname': host, 'cpu_model_name': cpu, 'os_release': 'RHEL 8.10', 'lsb_queue': 'standard'},
              open(os.path.join(pk_dir, f'{jobname}.{jid}_{idx}.json'), 'w'))


def matrices(seed):
    rng = np.random.default_rng(seed)
    x = rng.random((N, 4))
    d = np.abs(x[:, None, :] - x[None, :, :]).sum(-1)
    np.fill_diagonal(d, 1e6)
    un = pd.DataFrame(d, index=IDS, columns=IDS)
    m = d.copy()
    for i, a in enumerate(IDS):
        for j, b in enumerate(IDS):
            if i != j and PATIENT[a] == PATIENT[b]:
                m[i, j] = 1e6
    return un, pd.DataFrame(m, index=IDS, columns=IDS)


N_PAIRS = sum(1 for a in IDS for b in IDS if a < b and PATIENT[a] == PATIENT[b])
AGG_WIDTH = {'median_pooling': 1280, 'sampler': 12800}     # literal AMENDMENT_004 values (not read from G)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(F, 'N_SLIDES', N)
    base = str(tmp_path)
    fp32 = os.path.join(base, 'fp32')
    os.makedirs(fp32)
    for s in IDS:
        open(os.path.join(fp32, f'{s}.pkl'), 'w').write('')
    rd = os.path.join(base, 'pkg', 'run_001')
    fd = os.path.join(base, 'pkg', 'fidelity_001')
    os.makedirs(fd)
    pub = {}
    for k, m in enumerate(F.METHODS):
        un, ma = matrices(k)
        pub[m] = os.path.join(base, 'pub', f'{m}.pkl')
        pub[f'{m}_no_same_patient'] = os.path.join(base, 'pub', f'{m}_nsp.pkl')
        pk(pub[m], un)
        pk(pub[f'{m}_no_same_patient'], ma)
        un2 = un.copy()
        ma2 = ma.copy()
        live = un2.to_numpy() < 1e6
        un2.values[live] *= 1.0000001
        live = ma2.to_numpy() < 1e6
        ma2.values[live] *= 1.0000001
        pk(G.matrix_path(rd, m, False), un2)
        pk(G.matrix_path(rd, m, True), ma2)
    for m in G.AGG_METHODS:
        width = AGG_WIDTH[m]          # AMENDMENT_004: the stock aggregate_cli slide-vector width per method
        json.dump({'same_patient_pairs_masked': N_PAIRS, 'pass': True,
                   'provenance': {'method': m, 'n_slides': N, 'feature_dim': width}},
                  open(os.path.join(fd, f'm1_rebuild_{m}.json'), 'w'))
        os.makedirs(os.path.join(rd, 'logs', 'aggregate'), exist_ok=True)
        json.dump({'model': 'virchow2_fp32', 'n_slides': N, 'feature_dim': width, 'same_patient_pairs_masked': N_PAIRS},
                  open(os.path.join(rd, 'logs', 'aggregate', f'provenance_virchow2_fp32_{m}.json'), 'w'))
    for m in ('ot_cosine', 'yottixel'):
        os.makedirs(os.path.join(rd, 'logs', 'combine_similarity'), exist_ok=True)
        json.dump({'model': 'virchow2_fp32', 'n_query_slides': N, 'n_reference_slides': N, 'square': True,
                   'same_patient_pairs_masked': N_PAIRS},
                  open(os.path.join(rd, 'logs', 'combine_similarity', f'provenance_virchow2_fp32_{m}.json'), 'w'))
    for kind in ('ot', 'bob'):
        tdir = os.path.join(rd, 'similarity', f'virchow2_fp32_{kind}')
        for i, s in enumerate(IDS):
            pk(os.path.join(tdir, f'similarity_{s}.pkl'), [(o, 0.5 if o != s else 1e6) for o in IDS])
        pdir = os.path.join(rd, 'logs', f'similarity_virchow2_fp32_{kind}')
        os.makedirs(pdir)
        n_el = N if kind == 'ot' else G.n_bob_elements(N)
        canary = {'ot': (5, 7), 'bob': (2,)}[kind]
        for i in range(1, n_el + 1):
            rows = [i, i] if kind == 'ot' else [G.bob_rows(i, N)[0], G.bob_rows(i, N)[-1]]
            json.dump({'feature_dim': 1280, 'query_features': fp32, 'n_library': N, 'query_range': rows,
                       'encoder': 'virchow2_fp32', 'method': kind, 'external': False, 'num_workers': 1,
                       'lsf_index': str(i)}, open(os.path.join(pdir, f'provenance_{i}.json'), 'w'))
            jn = G.JOBS[kind][0] if i in canary else G.JOBS[kind][1]
            jid = 100 + i if i in canary else 900
            lsf_out(rd, jn, jid, i)
            lsf_err(rd, jn, jid, i)
            sidecar(rd, jn, jid, i, host=f'node{i % 3}')
    json.dump({'per_row': [{'row_pickle_bytes': os.path.getsize(
        os.path.join(rd, 'similarity', 'virchow2_fp32_ot', f'similarity_{IDS[0]}.pkl'))}]},
        open(os.path.join(fd, 'm1_ot_probe.json'), 'w'))
    return {'rd': rd, 'fd': fd, 'fp32': fp32, 'pub': pub}


def run(w):
    return G.m3_gates(w['rd'], w['fd'], w['fp32'], pub_paths=w['pub'], verify_published=False)


def test_all_pass(world):
    rep, census = run(world)
    assert rep['pass'], rep['failures'][:5]
    g = rep['gates']
    assert g['m3_5_1_rows']['ot_cosine']['n_provenance'] == N
    assert g['m3_5_1_rows']['yottixel']['n_provenance'] == 3
    assert g['m3_5_1_rows']['ot_cosine']['row_bytes_within_probe_range'] is True
    assert g['m3_5_2_3_matrices']['yottixel']['n_ge_1e6'] == N
    assert g['m3_5_2_3_matrices']['yottixel_no_same_patient']['n_ge_1e6'] == N + 2 * N_PAIRS
    d = g['m3_5_5_descriptive_vs_published']['sampler']
    assert d['max_rel'] == pytest.approx(1e-7, rel=1e-3) and d['median_abs'] > 0
    assert len([r for r in census if r['binding']]) == N + 3


def test_cmd_m3_writes_outputs_and_never_overwrites(world, monkeypatch):
    orig = G.m3_gates
    monkeypatch.setattr(G, 'm3_gates', lambda rd, fd, fp32: orig(rd, fd, fp32, pub_paths=world['pub'],
                                                                 verify_published=False))
    a = type('A', (), {'run_dir': world['rd'], 'fidelity_dir': world['fd'], 'fp32': world['fp32'],
                       'allow_run_dir_mismatch': True})          # ADDENDUM_002 F4: synthetic dir is named run_001
    assert G.cmd_m3(a) == 0
    for f in ('m3_matrix_gates.json', 'stderr_census.tsv', 'host_census.tsv', 'run_resources.json'):
        assert os.path.exists(os.path.join(world['rd'], f))
    hc = open(os.path.join(world['rd'], 'host_census.tsv')).read().splitlines()
    assert hc[0].split('\t')[:5] == ['job', 'job_id', 'index', 'hostname', 'cpu_model_name'] and len(hc) == N + 4
    rr = json.load(open(os.path.join(world['rd'], 'run_resources.json')))
    assert rr['per_job']['v2m_m3_ot_fanout']['n'] == N - 2
    with pytest.raises(G.GateError):
        G.cmd_m3(a)


def test_traceback_and_failed_line_stop(world):
    lsf_err(world['rd'], G.JOBS['ot'][1], 900, 9, 'Traceback (most recent call last):\n')
    lsf_out(world['rd'], G.JOBS['bob'][1], 900, 3, stdout='[bob] FAILED pair 1 2\n')
    rep, _ = run(world)
    assert not rep['pass']
    assert any('Traceback' in f for f in rep['failures']) and any('FAILED line' in f for f in rep['failures'])


def test_numitermax_reported_not_stop(world):
    lsf_err(world['rd'], G.JOBS['ot'][1], 900, 9, 'UserWarning: numItermax reached before optimality\n')
    rep, _ = run(world)
    assert rep['pass'] and rep['gates']['m3_5_1_rows']['ot_cosine']['numItermax_elements'][0][1] == 9


def test_retry_supersedes_and_superseded_traceback_not_binding(world):
    lsf_out(world['rd'], G.JOBS['ot'][1], 900, 11, rc=130)
    lsf_err(world['rd'], G.JOBS['ot'][1], 900, 11, 'Traceback x\n')
    rep, _ = run(world)
    assert not rep['pass']
    lsf_out(world['rd'], G.JOBS['ot'][1], 950, 11)
    lsf_err(world['rd'], G.JOBS['ot'][1], 950, 11)
    sidecar(world['rd'], G.JOBS['ot'][1], 950, 11)
    rep, census = run(world)
    assert rep['pass'], rep['failures'][:3]
    assert [r['binding'] for r in census if r['index'] == 11 and r['job'] == G.JOBS['ot'][1]] == [False, True]


def test_missing_element_row_provenance_sidecar(world):
    rd = world['rd']
    os.remove(os.path.join(rd, 'similarity', 'virchow2_fp32_ot', f'similarity_{IDS[3]}.pkl'))
    os.remove(os.path.join(rd, 'logs', 'similarity_virchow2_fp32_bob', 'provenance_2.json'))
    os.remove(os.path.join(rd, 'host_sidecar', f'{G.JOBS["ot"][1]}.900_12.json'))
    os.remove(os.path.join(rd, 'lsf', f'{G.JOBS["ot"][1]}.900_13.out'))
    rep, _ = run(world)
    f = ' | '.join(rep['failures'])
    assert 'row file ids != bank ids' in f and 'provenance files' in f and 'A9' in f and 'element 13' in f


def test_row_content_checks(world):
    p = os.path.join(world['rd'], 'similarity', 'virchow2_fp32_bob', f'similarity_{IDS[0]}.pkl')
    os.remove(p)
    pk(p, [(o, 0.1) for o in IDS[:-1]] + [(IDS[0], 0.1)])
    rep, _ = run(world)
    assert any('row id set' in f or 'pairs' in f for f in rep['failures'])


def test_sentinel_rules(world):
    rd = world['rd']
    un = pickle.load(open(G.matrix_path(rd, 'ot_cosine', False), 'rb'))
    os.remove(G.matrix_path(rd, 'ot_cosine', False))
    un.iloc[0, 1] = 1e6                               # a BoB-style per-pair failure off the diagonal
    pk(G.matrix_path(rd, 'ot_cosine', False), un)
    ma = pickle.load(open(G.matrix_path(rd, 'yottixel', True), 'rb'))
    os.remove(G.matrix_path(rd, 'yottixel', True))
    ma.iloc[0, 1] = 0.3                               # same-patient pair left unmasked
    pk(G.matrix_path(rd, 'yottixel', True), ma)
    rep, _ = run(world)
    f = rep['failures']
    assert 'M3.5.2/3 ot_cosine: sentinel_exactly_diagonal' in f
    assert 'M3.5.2/3 yottixel_no_same_patient: pattern_equal_published_masked' in f


def test_structure_rules(world):
    rd = world['rd']
    un = pickle.load(open(G.matrix_path(rd, 'sampler', False), 'rb'))
    os.remove(G.matrix_path(rd, 'sampler', False))
    un = un.astype('float32').iloc[::-1]
    pk(G.matrix_path(rd, 'sampler', False), un)
    rep, _ = run(world)
    f = rep['failures']
    assert 'M3.5.2/3 sampler: float64' in f and 'M3.5.2/3 sampler: index_equal_published_str' in f


def test_provenance_rules(world):
    rd = world['rd']
    p = os.path.join(rd, 'logs', 'combine_similarity', 'provenance_virchow2_fp32_yottixel.json')
    d = json.load(open(p))
    d['same_patient_pairs_masked'] = N_PAIRS + 1
    d['square'] = False
    json.dump(d, open(p, 'w'))
    p = os.path.join(rd, 'logs', 'similarity_virchow2_fp32_ot', 'provenance_4.json')
    d = json.load(open(p))
    d['query_features'] = '/elsewhere/tmp_final'
    json.dump(d, open(p, 'w'))
    rep, _ = run(world)
    f = ' | '.join(rep['failures'])
    assert 'yottixel: same_patient_pairs_masked_eq_rebuild' in f and 'yottixel: square_true' in f
    assert 'provenance_4.json: query_features' in f


def test_failed_elements_jsonl_stops(world):
    open(os.path.join(world['rd'], 'failed_elements.jsonl'), 'w').write('{"index": 3}\n')
    rep, _ = run(world)
    assert not rep['pass'] and 'failed_elements' in rep['failures'][0]


def test_m4_ranks_resolution(world):
    rd = world['rd']
    os.makedirs(os.path.join(rd, 'c2_metric_suite'))
    for m in F.METHODS:
        r = os.path.join(rd, 'c2_metric_suite', f'ranked_lists_virchow2_fp32_{m}.csv.gz')
        open(r, 'wb').write(b'x')
        json.dump({'input_similarity_matrix': G.matrix_path(rd, m, True), 'ranked_lists': r},
                  open(os.path.join(rd, 'c2_metric_suite', f'metrics_virchow2_fp32_{m}.json'), 'w'))
    rep = G.m4_ranks_resolution(rd)
    assert rep['pass'], rep['failures']
    json.dump({'input_similarity_matrix': os.path.join(F.PUBM, 'x.pkl'), 'ranked_lists': r},
              open(os.path.join(rd, 'c2_metric_suite', 'metrics_virchow2_fp32_yottixel.json'), 'w'))
    rep = G.m4_ranks_resolution(rd)
    assert not rep['pass'] and 'M4.1 yottixel: resolved_is_fp32_run_matrix' in rep['failures']


def test_tie_host_diversity(world, tmp_path):
    p = str(tmp_path / 'tie.csv')
    hdr = 'display_encoder,aggregation,cell_id,matrix_sha256,lineage,rank1_tie,' + ','.join(
        G.TIE_ORDER_COLS) + '\n'
    open(p, 'w').write(hdr + 'V,Y,virchow2_fp32/yottixel,x,p,1,0,3,0,0\nV,O,virchow2_fp32/ot_cosine,x,p,0,0,0,0,0\n'
                       'U,O,uni2/ot_cosine,x,p,0,9,9,9,9\n')
    rep = G.tie_host_diversity(world['rd'], p)
    assert rep['order_sensitive_cells'] == ['virchow2_fp32/yottixel']
    y = [c for c in rep['cells'] if c['cell_id'] == 'virchow2_fp32/yottixel'][0]
    assert y['n_elements_with_sidecar'] == 3 and y['n_hostnames'] == 3 and y['cpu_models'] == ['AMD EPYC 7713']


def _package_world(world, tmp_path, run_b=False):
    rd, fd = world['rd'], world['fd']
    open(os.path.join(fd, 'COMPLETE'), 'w').write('{}\n')
    for f in ('m3_matrix_gates.json', 'm4_ranks_resolution.json'):
        json.dump({'pass': True}, open(os.path.join(rd, f), 'w'))
    os.makedirs(os.path.join(rd, 'eval_attempt_001'))
    open(os.path.join(rd, 'eval_attempt_001', 'COMPLETE.json'), 'w').write('{}')
    ra = os.path.join(rd, 'rr5_additive_001')
    os.makedirs(ra)
    json.dump({'status': 'COMPLETE', 'spec_is_contract': False}, open(os.path.join(ra, 'COMPLETE.json'), 'w'))
    for f in ('bit_identity_report.json', 'evaluator_crosscheck.json'):
        json.dump({'pass': True}, open(os.path.join(ra, f), 'w'))
    os.makedirs(os.path.join(rd, 'comparison_001'))
    cr = _compare_report(run_b=None)
    if run_b:
        rb = os.path.join(rd, 'rr5_replacement_001')
        os.makedirs(os.path.join(rb, 'raw'))
        json.dump({'pass': True, 'oracle_winner_count_reproduces_submitted': {'value': False}},
                  open(os.path.join(rb, 'faithfulness_report.json'), 'w'))
        json.dump({'status': 'COMPLETE'}, open(os.path.join(rb, 'raw', 'COMPLETE.json'), 'w'))
        cr = _compare_report(run_b=os.path.join(rb, 'faithfulness_report.json'))
    json.dump(cr, open(os.path.join(rd, 'comparison_001', 'compare_report.json'), 'w'))
    json.dump({'package_flag': False}, open(os.path.join(rd, 'comparison_001', 'materiality.json'), 'w'))
    c = str(tmp_path / 'CONTRACT.md')
    open(c, 'w').write('contract')
    return c


def _compare_report(run_b=None, **over):
    """compare_report.json as do_compare writes it (only the keys package reads)."""
    r = {'pass': True, 'contract_draws': True, 'n_reps': 5000, 'draws_sha256': G.CONTRACT_DRAWS_SHA,
         'bootstrap_mode': G.CONTRACT_BOOTSTRAP_MODE}
    if run_b is None:
        r.update({'run_b': 'pending', 'a3': {'status': 'pending'}})
    else:
        r.update({'run_b': 'present', 'a3': {'pass': True},
                  'run_b_faithfulness_report': {'path': run_b, 'sha256': F.sha256_file(run_b)}})
    r.update(over)
    return r


def test_package_partial_without_run_b(world, tmp_path):
    """F4: Run B infeasible/not released -> PARTIAL.json (COMPLETE_RUN_A_ONLY, replacement products pending),
    never COMPLETE.json."""
    c = _package_world(world, tmp_path)
    sha = F.sha256_file(c)
    rep = G.package(world['rd'], world['fd'], contract=c, contract_sha=sha, run_b_status='infeasible')
    assert rep['status'] == 'COMPLETE_RUN_A_ONLY' and rep['run_b'] == 'infeasible'
    assert not os.path.exists(os.path.join(world['rd'], 'COMPLETE.json'))
    comp = json.load(open(os.path.join(world['rd'], 'PARTIAL.json')))
    assert comp['status'] == 'COMPLETE_RUN_A_ONLY' and comp['replacement_products'] == 'pending'
    assert comp['a3'] == 'pending' and comp['bootstrap_mode']['n_reps'] == 5000
    assert comp['contract']['sha256'] == sha and len(comp['matrices']) == 8
    assert comp['rr5_internal_spec_is_contract']['run_a'] is False
    sums = open(os.path.join(world['rd'], 'SHA256SUMS')).read()
    assert 'similarity/virchow2_fp32_ot.sha256' in sums and 'similarity/virchow2_fp32_ot/similarity_' not in sums
    assert 'logs/similarity_virchow2_fp32_ot/provenance_1.json' in sums
    man = open(os.path.join(world['rd'], 'similarity', 'virchow2_fp32_bob.sha256')).read().splitlines()
    assert len(man) == N
    assert comp['sha256sums_sha256'] == F.sha256_file(os.path.join(world['rd'], 'SHA256SUMS'))
    assert 'PARTIAL.json' not in sums
    with pytest.raises(G.GateError, match='PARTIAL.json already exists'):
        G.package(world['rd'], world['fd'], contract=c, contract_sha=sha, run_b_status='infeasible')
    assert G.main(['package', '--run-dir', world['rd'], '--fidelity-dir', world['fd'],
                   '--run-b-status', 'infeasible']) == 1


def test_package_complete_with_run_b(world, tmp_path):
    c = _package_world(world, tmp_path, run_b=True)
    rep = G.package(world['rd'], world['fd'], contract=c, contract_sha=F.sha256_file(c))
    assert rep['status'] == 'COMPLETE' and rep['run_b'] == 'COMPLETE' and rep['a3'] == 'pass'
    assert os.path.exists(os.path.join(world['rd'], 'COMPLETE.json'))
    assert not os.path.exists(os.path.join(world['rd'], 'PARTIAL.json'))


def test_f1_p3_compare_pending_then_run_b_appears(world, tmp_path):
    """Review probe P3: compare ran with Run B pending; Run B then appears -> package must not write COMPLETE."""
    c = _package_world(world, tmp_path)
    rb = os.path.join(world['rd'], 'rr5_replacement_001')
    os.makedirs(os.path.join(rb, 'raw'))
    json.dump({'pass': True}, open(os.path.join(rb, 'faithfulness_report.json'), 'w'))
    json.dump({'status': 'COMPLETE'}, open(os.path.join(rb, 'raw', 'COMPLETE.json'), 'w'))
    rep = G.package(world['rd'], world['fd'], contract=c, contract_sha=F.sha256_file(c))
    assert rep['status'] == 'FAILED' and any('comparison_001_report' in f for f in rep['failures'])
    assert not os.path.exists(os.path.join(world['rd'], 'COMPLETE.json'))
    assert not os.path.exists(os.path.join(world['rd'], 'PARTIAL.json'))


@pytest.mark.parametrize('over,with_b', [
    ({'n_reps': 200}, False), ({'contract_draws': False}, False), ({'draws_sha256': '0' * 64}, False),
    ({'bootstrap_mode': 'whole-patient multiplicity weights, n_reps = 200, draws sha256 x (NOT draws #46)'}, False),
    ({'a3': {'pass': False}}, True), ({'run_b': 'pending', 'a3': {'status': 'pending'}}, True),
    ({'run_b_faithfulness_report': {'sha256': '0' * 64}}, True), ({'run_b': 'present'}, False)])
def test_f1_f15_compare_report_must_agree(world, tmp_path, over, with_b):
    c = _package_world(world, tmp_path, run_b=with_b)
    p = os.path.join(world['rd'], 'comparison_001', 'compare_report.json')
    r = json.load(open(p))
    r.update(over)
    json.dump(r, open(p, 'w'))
    rep = G.package(world['rd'], world['fd'], contract=c, contract_sha=F.sha256_file(c),
                    run_b_status=None if with_b else 'not_released')
    assert rep['status'] == 'FAILED' and any('comparison_001_report' in f for f in rep['failures'])
    for name in ('COMPLETE.json', 'PARTIAL.json'):
        assert not os.path.exists(os.path.join(world['rd'], name))


def test_package_refuses_run_b_status_with_run_b(world, tmp_path):
    c = _package_world(world, tmp_path, run_b=True)
    rep = G.package(world['rd'], world['fd'], contract=c, contract_sha=F.sha256_file(c), run_b_status='infeasible')
    assert rep['status'] == 'FAILED' and any('run-b-status' in f for f in rep['failures'])


def test_contract_bootstrap_constants_match_compare_and_adapter():
    import importlib.util as iu
    sp = iu.spec_from_file_location('v2main_compare_for_gates_test',
                                    os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_compare.py'))
    C = iu.module_from_spec(sp)
    sp.loader.exec_module(C)
    assert G.CONTRACT_BOOTSTRAP_MODE == C.BOOTSTRAP_MODE and G.CONTRACT_DRAWS_SHA == C.A.DRAWS_SHA


def test_package_failed_on_missing_stage_and_run_b_status(world, tmp_path):
    c = _package_world(world, tmp_path)
    os.remove(os.path.join(world['rd'], 'm4_ranks_resolution.json'))
    rep = G.package(world['rd'], world['fd'], contract=c, contract_sha=F.sha256_file(c))
    assert rep['status'] == 'FAILED'
    assert any('m4_ranks_resolution' in f for f in rep['failures'])
    assert any('Run B absent' in f for f in rep['failures'])
    assert os.path.exists(os.path.join(world['rd'], 'FAILED'))
    assert not os.path.exists(os.path.join(world['rd'], 'COMPLETE.json'))


def test_package_refuses_wrong_contract(world, tmp_path):
    c = _package_world(world, tmp_path)
    with pytest.raises(G.GateError, match='contract'):
        G.package(world['rd'], world['fd'], contract=c, contract_sha='0' * 64, run_b_status='infeasible')


def _fake_contract(tmp_path, monkeypatch, files):
    """A synthetic §I table covering each path form the parser resolves."""
    import hashlib
    ps, repo, out, att, pubm = (str(tmp_path / d) for d in ('ps', 'repo', 'out', 'att', 'pubm'))
    for d in (ps, repo, out, att, pubm):
        os.makedirs(d, exist_ok=True)
    monkeypatch.setattr(G, 'PS', ps)
    monkeypatch.setattr(F, 'REPO', repo)
    monkeypatch.setattr(F, 'OUT', out)
    monkeypatch.setattr(G, 'ATTEMPT', att)
    monkeypatch.setattr(F, 'PUBM', pubm)
    real = {}
    for n, base, rel, content in files:
        p = os.path.join({'PS': ps, 'REPO': repo, 'OUT': out, 'ATT': att, 'PUBM': pubm, 'ABS': str(tmp_path)}[base],
                         rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'wb').write(content)
        real[n] = (p, hashlib.sha256(content).hexdigest())
    spec = os.path.join(out, 'spec43.json')
    json.dump({'sources': {'metadata': {'path': real[57][0], 'sha256': real[57][1]}}}, open(spec, 'w'))
    real[43] = (spec, F.sha256_file(spec))
    rows = [f'| 1 | `PS/prosper/a.py` | `{real[1][1]}` | C |',
            f'| 24 | `REPO/scripts/b.py` (RR-5) | `{real[24][1]}` | C = R |',
            f'| 35 | `m.pkl` | 68,294,649 | `{real[35][1]}` | C |',
            f'| 43 | `OUT/spec43.json` (template) | `{real[43][1]}` | C |',
            f'| 45 | `\u2026/main_virchow2_attempt_001/populations.csv` | `{real[45][1]}` | C = R |',
            f'| 57 | metadata (spec source `metadata`, 3 B; path as in #43) | `{real[57][1]}` | R (#43) |',
            '| 70 | UNI2 qm | uni2 qm `7311b6c0\u20269b5e` | R |']
    c = str(tmp_path / 'CONTRACT.md')
    open(c, 'w').write('### I.1\n\n| # | file | sha256 | src |\n|---|---|---|---|\n' + '\n'.join(rows) + '\n')
    return c, real


def test_input_bindings_contract_parse_and_rehash(tmp_path, monkeypatch):
    c, real = _fake_contract(tmp_path, monkeypatch, [(1, 'PS', 'prosper/a.py', b'a'), (24, 'REPO', 'scripts/b.py', b'b'),
                                                     (35, 'PUBM', 'm.pkl', b'm'), (45, 'ATT', 'populations.csv', b'p'),
                                                     (57, 'ABS', 'meta.csv', b'md')])
    sha = F.sha256_file(c)
    t = G.contract_inputs(c, sha)
    assert sorted(t) == [1, 24, 35, 43, 45, 57] and all(t[n][0] == real[n][0] for n in t)
    rep = G.input_bindings('run_001', c, sha, numbers=[1, 24, 35, 43, 45, 57])
    assert rep['pass'] and len(rep['inputs']) == 6 and all(r['match'] for r in rep['inputs'])
    open(real[45][0], 'wb').write(b'changed')
    rep = G.input_bindings('run_001', c, sha, numbers=[1, 45, 70])
    assert not rep['pass'] and rep['failures'] == ['M0.3 #45: sha256 mismatch ' + real[45][0],
                                                   '#70: no full sha256 row in the contract']
    with pytest.raises(G.GateError):
        G.contract_inputs(c, '0' * 64)


def test_input_bindings_cmd_writes_failed(tmp_path, monkeypatch):
    c, real = _fake_contract(tmp_path, monkeypatch, [(1, 'PS', 'prosper/a.py', b'a'), (24, 'REPO', 'scripts/b.py', b'b'),
                                                     (35, 'PUBM', 'm.pkl', b'm'), (45, 'ATT', 'populations.csv', b'p'),
                                                     (57, 'ABS', 'meta.csv', b'md')])
    monkeypatch.setattr(G, 'CONTRACT', c)
    monkeypatch.setattr(G, 'CONTRACT_SHA', F.sha256_file(c))
    monkeypatch.setattr(G, 'STAGE_INPUTS', {'run_001': [1, 24], 'fidelity_001': [1, 35, 99]})
    orig = G.input_bindings
    monkeypatch.setattr(G, 'input_bindings', lambda st: orig(st, c, F.sha256_file(c)))
    sd = str(tmp_path / 'stage_ok')
    assert G.main(['input-bindings', '--stage', 'run_001', '--stage-dir', sd]) == 0
    assert json.load(open(os.path.join(sd, 'input_bindings.json')))['pass'] is True
    sd2 = str(tmp_path / 'stage_bad')
    assert G.main(['input-bindings', '--stage', 'fidelity_001', '--stage-dir', sd2]) == 1
    assert os.path.exists(os.path.join(sd2, 'FAILED'))
    assert G.main(['input-bindings', '--stage', 'run_001', '--stage-dir', sd]) == 1      # never overwrite


# ----------------------------------------------------------------------------------------------
# AMENDMENT_003: fidelity_002 default and the renamed `fidelity_COMPLETE` package entry
# ----------------------------------------------------------------------------------------------

def test_a003_package_records_fidelity_complete_with_dir(world, tmp_path):
    c = _package_world(world, tmp_path)
    rep = G.package(world['rd'], world['fd'], contract=c, contract_sha=F.sha256_file(c), run_b_status='infeasible')
    so = json.load(open(os.path.join(world['rd'], 'PARTIAL.json')))['stage_outcomes']
    assert rep['status'] == 'COMPLETE_RUN_A_ONLY'
    assert 'fidelity_001_COMPLETE' not in so
    e = so['fidelity_COMPLETE']
    assert e['ok'] and e['fidelity_dir'] == world['fd'] and e['path'] == os.path.join(world['fd'], 'COMPLETE')


def test_a003_package_inputs_label_on_missing_complete(tmp_path):
    fd = str(tmp_path / 'fidelity_002')
    os.makedirs(fd)
    rec, fails = G.package_inputs(str(tmp_path / 'run_001'), fd)
    assert rec['fidelity_COMPLETE'] == {'path': os.path.join(fd, 'COMPLETE'), 'present': False, 'ok': False,
                                        'fidelity_dir': fd}
    assert fails[0] == f"fidelity_COMPLETE: {os.path.join(fd, 'COMPLETE')}"
    open(os.path.join(fd, 'FAILED'), 'w').write('x')
    assert f'{fd}: FAILED present' in G.package_inputs(str(tmp_path / 'run_001'), fd)[1]


def test_a003_default_fidelity_dir_is_fidelity_002(monkeypatch):
    assert F.FIDELITY_DIR.endswith(os.path.join('r2_1_virchow2_fp32_main_20261013', 'fidelity_002'))
    seen = {}
    for fn in ('cmd_m3', 'cmd_m4_ranks', 'cmd_tie_hosts', 'cmd_package'):
        monkeypatch.setattr(G, fn, (lambda n: lambda a: seen.__setitem__(n, a) or 0)(fn))
    for sub in ('m3', 'm4-ranks', 'tie-hosts', 'package'):
        assert G.main([sub]) == 0
    assert set(seen) == {'cmd_m3', 'cmd_m4_ranks', 'cmd_tie_hosts', 'cmd_package'}
    assert all(a.fidelity_dir == F.FIDELITY_DIR for a in seen.values())


# ----------------------------------------------------------------------------------------------
# AMENDMENT_004: per-method aggregate feature_dim; run_002 lineage bindings
# ----------------------------------------------------------------------------------------------

def _set_agg_dim(world, m, width=None, rebuild=None):
    if width is not None:
        p = os.path.join(world['rd'], 'logs', 'aggregate', f'provenance_virchow2_fp32_{m}.json')
        d = json.load(open(p))
        d['feature_dim'] = width
        json.dump(d, open(p, 'w'))
    if rebuild is not None:
        p = os.path.join(world['fd'], f'm1_rebuild_{m}.json')
        d = json.load(open(p))
        d['provenance']['feature_dim'] = rebuild
        json.dump(d, open(p, 'w'))


def test_a004_constants_and_defaults():
    assert G.AGG_FEATURE_DIM == {'median_pooling': 1280, 'sampler': 12800} and F.WIDTH == 1280
    # ADDENDUM_003 (implementation_008): the run-stage default moved run_002 -> run_003
    assert G.RUN_DIR == F.RUN_DIR_003 and G.RUN_DIR.endswith(os.path.join('r2_1_virchow2_fp32_main_20261013',
                                                                           'run_003'))
    assert G.STAGE_INPUTS['run_002'] == G.STAGE_INPUTS['run_001'] == G.STAGE_INPUTS['run_003']
    c = G.provenance_checks('sampler', {'feature_dim': 12800, 'n_slides': N, 'model': G.MODEL,
                                        'same_patient_pairs_masked': 1}, N, 1, 12800)
    assert 'feature_dim_1280' not in c and all(c.values())
    assert set(c) == {'same_patient_pairs_masked_eq_rebuild', 'n_slides_eq_n', 'feature_dim_matches_method',
                      'fidelity_rebuild_feature_dim_consistent', 'model_is_fp32'}


def test_a004_sampler_and_median_feature_dim_pass_recorded(world):
    rep, _ = run(world)
    assert rep['pass'], rep['failures'][:5]
    g4 = rep['gates']['m3_5_4_provenance']
    for m in G.AGG_METHODS:
        assert g4[m]['feature_dim_expected'] == g4[m]['feature_dim_observed'] == AGG_WIDTH[m]
        assert g4[m]['feature_dim_matches_method'] is True and g4[m]['fidelity_rebuild_feature_dim'] == AGG_WIDTH[m]
        assert g4[m]['checks']['fidelity_rebuild_feature_dim_consistent'] is True
        assert g4[m]['fidelity_rebuild_path'] == os.path.join(world['fd'], f'm1_rebuild_{m}.json')
    assert rep['gates']['rebuild_feature_dim'] == AGG_WIDTH
    for k in ('amendment_004', 'run_001_failed', 'lineage_file'):
        assert k not in rep                                   # no LINEAGE_RUN_002.json in this run dir


@pytest.mark.parametrize('m,width', [('sampler', 1280), ('median_pooling', 12800)])
def test_a004_wrong_method_width_fails(world, m, width):
    _set_agg_dim(world, m, width=width, rebuild=width)        # rebuild agrees: only the method expectation fails
    rep, _ = run(world)
    assert not rep['pass']
    assert rep['failures'] == [f'M3.5.3/4 {m}: feature_dim_matches_method']
    assert rep['first_failing_gate'] == f'M3.5.3/4 {m}: feature_dim_matches_method'
    g = rep['gates']['m3_5_4_provenance'][m]
    assert (g['feature_dim_expected'], g['feature_dim_observed'], g['feature_dim_matches_method']) == (
        AGG_WIDTH[m], width, False)


@pytest.mark.parametrize('m', ['sampler', 'median_pooling'])
def test_a004_fidelity_rebuild_feature_dim_mismatch_fails(world, m):
    _set_agg_dim(world, m, rebuild=AGG_WIDTH[m] + 1)
    rep, _ = run(world)
    assert rep['failures'] == [f'M3.5.3/4 {m}: fidelity_rebuild_feature_dim_consistent']
    p = os.path.join(world['fd'], f'm1_rebuild_{m}.json')
    d = json.load(open(p))
    del d['provenance']                                       # unrecorded rebuild width: not consistent
    json.dump(d, open(p, 'w'))
    rep, _ = run(world)
    assert rep['failures'] == [f'M3.5.3/4 {m}: fidelity_rebuild_feature_dim_consistent']


def test_a004_missing_observed_feature_dim_fails_both(world):
    p = os.path.join(world['rd'], 'logs', 'aggregate', 'provenance_virchow2_fp32_sampler.json')
    d = json.load(open(p))
    del d['feature_dim']
    json.dump(d, open(p, 'w'))
    rep, _ = run(world)
    assert rep['failures'] == ['M3.5.3/4 sampler: feature_dim_matches_method',
                               'M3.5.3/4 sampler: fidelity_rebuild_feature_dim_consistent']


def _a4_run002(world, tmp_path, monkeypatch):
    """Turn the world's run dir into a FAILED run_001 (with an old m3_gates attempt) and build run_002 from it with
    the real init_run_002."""
    rd1 = world['rd']
    open(os.path.join(rd1, 'FAILED'), 'w').write('{"first_failing_gate": "M3.5.3/4 sampler: feature_dim_1280"}\n')
    open(os.path.join(rd1, 'journal.jsonl'), 'w').write('{"event": "submit", "step": "m3_gates"}\n')
    lsf_out(rd1, 'v2m_m3_gates', 555, 0, rc=1)
    lsf_err(rd1, 'v2m_m3_gates', 555, 0, 'STOP: M3.5.3/4 sampler: feature_dim_1280\n')
    sidecar(rd1, 'v2m_m3_gates', 555, 0, host='nodeOld')
    for f in ('m3_matrix_gates.json', 'stderr_census.tsv', 'host_census.tsv', 'run_resources.json'):
        open(os.path.join(rd1, f), 'w').write('old run_001 output')
    amend = os.path.join(str(tmp_path), 'contract_amendment_004', 'AMENDMENT_004.md')
    os.makedirs(os.path.dirname(amend))
    open(amend, 'w').write('# amendment 004 (synthetic)\n')
    sha = F.sha256_file(amend)
    monkeypatch.setattr(F, 'AMENDMENT_004_SHA', sha)
    rd2 = os.path.join(os.path.dirname(rd1), 'run_002')
    F.init_run_002(rd1, rd2, amend, amendment_sha=sha)
    return rd1, rd2, amend, sha


def test_a004_run_002_lineage_bindings_and_census(world, tmp_path, monkeypatch):
    rd1, rd2, amend, sha = _a4_run002(world, tmp_path, monkeypatch)
    rep, census = G.m3_gates(rd2, world['fd'], world['fp32'], pub_paths=world['pub'], verify_published=False,
                             amendment=amend)
    assert rep['pass'], rep['failures'][:5]
    assert rep['amendment_004'] == {'path': amend, 'sha256': sha}
    assert rep['run_001_failed'] == {'path': os.path.join(rd1, 'FAILED'),
                                     'sha256': F.sha256_file(os.path.join(rd1, 'FAILED'))}
    assert rep['lineage_file'] == {'path': os.path.join(rd2, F.LINEAGE_RUN_002),
                                   'sha256': F.sha256_file(os.path.join(rd2, F.LINEAGE_RUN_002))}
    assert len([r for r in census if r['binding']]) == N + 3   # fan-out logs (hardlinked) still covered
    assert open(os.path.join(rd1, 'm3_matrix_gates.json')).read() == 'old run_001 output'   # run_001 untouched


def test_a004_cmd_m3_amendment_arg_records_bindings(world, tmp_path, monkeypatch):
    rd1, rd2, amend, sha = _a4_run002(world, tmp_path, monkeypatch)
    monkeypatch.setattr(G, 'RUN_STAGE', 'run_002')     # implementation_008: AMENDMENT_004-era run_002 m3 behaviour
    orig = G.m3_gates
    monkeypatch.setattr(G, 'm3_gates', lambda rd, fd, fp32, amendment=None: orig(
        rd, fd, fp32, pub_paths=world['pub'], verify_published=False, amendment=amendment))
    a = type('A', (), {'run_dir': rd2, 'fidelity_dir': world['fd'], 'fp32': world['fp32'], 'amendment': amend})
    assert G.cmd_m3(a) == 0
    out = json.load(open(os.path.join(rd2, 'm3_matrix_gates.json')))
    assert out['pass'] and out['amendment_004'] == {'path': amend, 'sha256': sha}
    assert out['run_001_failed']['sha256'] == F.sha256_file(os.path.join(rd1, 'FAILED'))
    assert not os.path.exists(os.path.join(rd2, 'FAILED'))
    rr = json.load(open(os.path.join(rd2, 'run_resources.json')))   # old m3_gates attempt excluded by init-run-002
    assert rr['per_job']['v2m_m3_ot_fanout']['n'] == N - 2 and 'v2m_m3_gates' not in rr['per_job']
    hc = open(os.path.join(rd2, 'host_census.tsv')).read()
    assert 'nodeOld' not in hc and len(hc.splitlines()) == N + 4
    assert open(os.path.join(rd1, 'm3_matrix_gates.json')).read() == 'old run_001 output'   # run_001 untouched
    assert F.check_lineage(rd2, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)['pass']
    assert not F.check_lineage(rd2, F.LINEAGE_RUN_002)['pass']      # strict mode: regenerated outputs present
    with pytest.raises(G.GateError):                           # write_new refusal semantics unchanged
        G.cmd_m3(a)


def test_a004_lineage_binding_refusals(world, tmp_path, monkeypatch):
    rd1, rd2, amend, sha = _a4_run002(world, tmp_path, monkeypatch)
    args = (world['fd'], world['fp32'])
    kw = {'pub_paths': world['pub'], 'verify_published': False}
    monkeypatch.setattr(F, 'AMENDMENT_004_SHA', '0' * 64)      # bound amendment sha differs
    with pytest.raises(G.GateError, match='AMENDMENT_004 sha256'):
        G.m3_gates(rd2, *args, amendment=amend, **kw)
    monkeypatch.setattr(F, 'AMENDMENT_004_SHA', sha)
    with pytest.raises(G.GateError, match='missing'):
        G.m3_gates(rd2, *args, amendment=amend + '.absent', **kw)
    open(os.path.join(rd1, 'FAILED'), 'a').write(' ')           # frozen run_001/FAILED changed
    with pytest.raises(G.GateError, match='run_001 FAILED sha256'):
        G.m3_gates(rd2, *args, amendment=amend, **kw)
    os.remove(os.path.join(rd1, 'FAILED'))
    with pytest.raises(G.GateError, match='FAILED .* missing'):
        G.m3_gates(rd2, *args, amendment=amend, **kw)
    os.remove(os.path.join(rd2, F.LINEAGE_RUN_002))            # a run dir named run_002 needs the lineage file
    with pytest.raises(G.GateError, match='init-run-002'):
        G.m3_gates(rd2, *args, amendment=amend, **kw)
    assert G.main(['m3', '--run-dir', rd2, '--fidelity-dir', world['fd'], '--fp32', world['fp32'],
                   '--amendment', amend]) == 1
    for f in ('m3_matrix_gates.json', 'FAILED', 'stderr_census.tsv'):
        assert not os.path.exists(os.path.join(rd2, f))          # STOP before anything is written
    open(os.path.join(rd2, F.LINEAGE_RUN_002), 'w').write('{not json')
    with pytest.raises(G.GateError, match='unreadable'):
        G.m3_gates(rd2, *args, amendment=amend, **kw)


def test_a004_m3_cli_amendment_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(G, 'cmd_m3', lambda a: seen.__setitem__('a', a) or 0)
    assert G.main(['m3']) == 0
    a = seen['a']
    assert a.amendment == F.AMENDMENT_004 and a.run_dir == F.RUN_DIR_003 and a.fidelity_dir == F.FIDELITY_DIR


# ----------------------------------------------------------------------------------------------
# ADDENDUM_002 F4: run-dir guard on m3 / m4-ranks / tie-hosts; F8: malformed fidelity rebuild record
# ----------------------------------------------------------------------------------------------

M3_OUTPUTS = ('m3_matrix_gates.json', 'stderr_census.tsv', 'host_census.tsv', 'run_resources.json', 'FAILED')


def _m3_args(world, rd, **kw):
    return type('A', (), {'run_dir': rd, 'fidelity_dir': world['fd'], 'fp32': world['fp32'], **kw})


def _patch_m3(world, monkeypatch):
    orig = G.m3_gates
    monkeypatch.setattr(G, 'm3_gates', lambda rd, fd, fp32, amendment=None: orig(
        rd, fd, fp32, pub_paths=world['pub'], verify_published=False, amendment=amendment))


def _ranks_world(rd):
    os.makedirs(os.path.join(rd, 'c2_metric_suite'), exist_ok=True)
    for m in F.METHODS:
        r = os.path.join(rd, 'c2_metric_suite', f'ranked_lists_virchow2_fp32_{m}.csv.gz')
        open(r, 'wb').write(b'x')
        json.dump({'input_similarity_matrix': G.matrix_path(rd, m, True), 'ranked_lists': r},
                  open(os.path.join(rd, 'c2_metric_suite', f'metrics_virchow2_fp32_{m}.json'), 'w'))


def _tie_csv(tmp_path):
    p = str(tmp_path / 'tie.csv')
    open(p, 'w').write('display_encoder,aggregation,cell_id,matrix_sha256,lineage,rank1_tie,' +
                       ','.join(G.TIE_ORDER_COLS) + '\nV,Y,virchow2_fp32/yottixel,x,p,1,0,3,0,0\n')
    return p


def _a8_run003(rd2, tmp_path, monkeypatch, **kw):
    """implementation_008: put FAILED into run_002 (if absent) and build run_003 from it with the real
    init_run_003 and a synthetic ADDENDUM_003."""
    if not os.path.exists(os.path.join(rd2, 'FAILED')):
        open(os.path.join(rd2, 'FAILED'), 'w').write('{"first_failing_gate": "M4.1 median_pooling: x"}\n')
    add = os.path.join(str(tmp_path), 'contract_amendment_004', 'ADDENDUM_003.md')
    os.makedirs(os.path.dirname(add), exist_ok=True)
    if not os.path.exists(add):
        open(add, 'w').write('# addendum 003 (synthetic)\n')
    monkeypatch.setattr(F, 'ADDENDUM_003_SHA', F.sha256_file(add))
    rd3 = os.path.join(os.path.dirname(rd2), 'run_003')
    F.init_run_003(rd2, rd3, add, addendum_sha=F.sha256_file(add), **kw)
    return rd3, add


def test_f4_run_stage_constant_mirrors_submit():
    assert G.RUN_STAGE == 'run_003' == os.path.basename(G.RUN_DIR)       # ADDENDUM_003
    src = open(os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_submit.py')).read()
    assert "RUN_STAGE = 'run_003'" in src                     # literal mirror, submit.py is not imported


def test_f4_m3_refuses_run_001_basename_without_flag(world, monkeypatch):
    _patch_m3(world, monkeypatch)
    rd = world['rd']                                          # .../pkg/run_001
    with pytest.raises(G.GateError, match='not the run_003 stage'):
        G.cmd_m3(_m3_args(world, rd))
    assert G.main(['m3', '--run-dir', rd, '--fidelity-dir', world['fd'], '--fp32', world['fp32']]) == 1
    for f in M3_OUTPUTS:
        assert not os.path.exists(os.path.join(rd, f))         # nothing written


def test_f4_m3_refuses_symlink_resolving_to_other_stage(world, monkeypatch, tmp_path):
    _patch_m3(world, monkeypatch)
    alias = os.path.join(str(tmp_path), 'alias', 'run_003')
    os.makedirs(os.path.dirname(alias))
    os.symlink(world['rd'], alias)                            # named run_003, resolves to run_001
    with pytest.raises(G.GateError, match='not the run_003 stage'):
        G.cmd_m3(_m3_args(world, alias))
    for f in M3_OUTPUTS:
        assert not os.path.exists(os.path.join(world['rd'], f))


def test_f4_m3_flag_override_records_guard(world, monkeypatch):
    _patch_m3(world, monkeypatch)
    rd = world['rd']
    assert G.main(['m3', '--run-dir', rd, '--fidelity-dir', world['fd'], '--fp32', world['fp32'],
                   '--allow-run-dir-mismatch']) == 0
    g = json.load(open(os.path.join(rd, 'm3_matrix_gates.json')))['run_dir_guard']
    assert g == {'run_dir': rd, 'run_dir_resolved': os.path.realpath(rd), 'run_stage': 'run_003',
                 'run_dir_mismatch_allowed': True}


def test_f4_m3_refuses_failed_run_dir_even_with_flag(world, monkeypatch):
    _patch_m3(world, monkeypatch)
    rd = world['rd']
    open(os.path.join(rd, 'FAILED'), 'w').write('{}')
    with pytest.raises(G.GateError, match='already holds FAILED'):
        G.cmd_m3(_m3_args(world, rd, allow_run_dir_mismatch=True))
    for f in M3_OUTPUTS[:-1]:
        assert not os.path.exists(os.path.join(rd, f))


def test_f4_m3_run_002_records_guard_and_failed_refusal(world, tmp_path, monkeypatch):
    rd1, rd2, amend, sha = _a4_run002(world, tmp_path, monkeypatch)
    _patch_m3(world, monkeypatch)
    with pytest.raises(G.GateError, match='not the run_003 stage'):      # ADDENDUM_003: run_002 is now a source
        G.cmd_m3(_m3_args(world, rd2, amendment=amend))
    monkeypatch.setattr(G, 'RUN_STAGE', 'run_002')     # implementation_008: AMENDMENT_004-era run_002 m3 behaviour
    assert G.cmd_m3(_m3_args(world, rd2, amendment=amend)) == 0
    g = json.load(open(os.path.join(rd2, 'm3_matrix_gates.json')))['run_dir_guard']
    assert g['run_stage'] == 'run_002' and g['run_dir_resolved'] == os.path.realpath(rd2)
    assert g['run_dir_mismatch_allowed'] is False
    for f in M3_OUTPUTS[:-1]:
        os.remove(os.path.join(rd2, f))
    open(os.path.join(rd2, 'FAILED'), 'w').write('{}')
    with pytest.raises(G.GateError, match='already holds FAILED'):
        G.cmd_m3(_m3_args(world, rd2, amendment=amend))
    for f in M3_OUTPUTS[:-1]:
        assert not os.path.exists(os.path.join(rd2, f))


def test_f4_m4_ranks_guard(world, tmp_path, monkeypatch):
    rd1 = world['rd']
    _ranks_world(rd1)
    assert G.main(['m4-ranks', '--run-dir', rd1]) == 1
    assert not os.path.exists(os.path.join(rd1, 'm4_ranks_resolution.json'))
    assert G.main(['m4-ranks', '--run-dir', rd1, '--allow-run-dir-mismatch']) == 0
    g = json.load(open(os.path.join(rd1, 'm4_ranks_resolution.json')))['run_dir_guard']
    assert g['run_dir_mismatch_allowed'] is True and g['run_stage'] == 'run_003'
    os.remove(os.path.join(rd1, 'm4_ranks_resolution.json'))
    shutil.rmtree(os.path.join(rd1, 'c2_metric_suite'))      # later stages get their own (not hardlinked) ranks files
    rd1, rd2, amend, sha = _a4_run002(world, tmp_path, monkeypatch)   # run_001 now holds FAILED
    assert G.main(['m4-ranks', '--run-dir', rd1, '--allow-run-dir-mismatch']) == 1
    assert not os.path.exists(os.path.join(rd1, 'm4_ranks_resolution.json'))
    _ranks_world(rd2)
    assert G.main(['m4-ranks', '--run-dir', rd2]) == 1           # ADDENDUM_003: run_002 is not the run stage
    assert not os.path.exists(os.path.join(rd2, 'm4_ranks_resolution.json'))
    shutil.rmtree(os.path.join(rd2, 'c2_metric_suite'))
    rd3, add = _a8_run003(rd2, tmp_path, monkeypatch)            # run_002 now holds FAILED
    assert G.main(['m4-ranks', '--run-dir', rd2, '--allow-run-dir-mismatch']) == 1
    _ranks_world(rd3)
    assert G.main(['m4-ranks', '--run-dir', rd3]) == 0
    g = json.load(open(os.path.join(rd3, 'm4_ranks_resolution.json')))['run_dir_guard']
    assert g == {'run_dir': rd3, 'run_dir_resolved': os.path.realpath(rd3), 'run_stage': 'run_003',
                 'run_dir_mismatch_allowed': False}
    os.remove(os.path.join(rd3, 'm4_ranks_resolution.json'))
    open(os.path.join(rd3, 'FAILED'), 'w').write('{}')
    assert G.main(['m4-ranks', '--run-dir', rd3, '--allow-run-dir-mismatch']) == 1
    assert not os.path.exists(os.path.join(rd3, 'm4_ranks_resolution.json'))


def test_f4_tie_hosts_guard(world, tmp_path):
    p = _tie_csv(tmp_path)
    rd1 = world['rd']
    assert G.main(['tie-hosts', '--run-dir', rd1, '--tie-census', p]) == 1
    assert not os.path.exists(os.path.join(rd1, 'tie_census_host_diversity.json'))
    assert G.main(['tie-hosts', '--run-dir', rd1, '--tie-census', p, '--allow-run-dir-mismatch']) == 0
    g = json.load(open(os.path.join(rd1, 'tie_census_host_diversity.json')))['run_dir_guard']
    assert g['run_dir_mismatch_allowed'] is True
    old = os.path.join(str(tmp_path), 'x', 'run_002')          # ADDENDUM_003: run_002 is a frozen source now
    os.makedirs(old)
    assert G.main(['tie-hosts', '--run-dir', old, '--tie-census', p]) == 1
    assert not os.path.exists(os.path.join(old, 'tie_census_host_diversity.json'))
    rd2 = os.path.join(str(tmp_path), 'y', 'run_003')
    os.makedirs(rd2)
    assert G.main(['tie-hosts', '--run-dir', rd2, '--tie-census', p]) == 0
    g = json.load(open(os.path.join(rd2, 'tie_census_host_diversity.json')))['run_dir_guard']
    assert g['run_stage'] == 'run_003' and g['run_dir_mismatch_allowed'] is False
    os.remove(os.path.join(rd2, 'tie_census_host_diversity.json'))
    open(os.path.join(rd2, 'FAILED'), 'w').write('{}')
    assert G.main(['tie-hosts', '--run-dir', rd2, '--tie-census', p]) == 1
    assert not os.path.exists(os.path.join(rd2, 'tie_census_host_diversity.json'))


def test_f4_cli_flag_defaults(monkeypatch):
    seen = {}
    for fn in ('cmd_m3', 'cmd_m4_ranks', 'cmd_tie_hosts', 'cmd_package'):
        monkeypatch.setattr(G, fn, (lambda n: lambda a: seen.__setitem__(n, a) or 0)(fn))
    for sub in ('m3', 'm4-ranks', 'tie-hosts'):
        assert G.main([sub]) == 0 and G.main([sub, '--allow-run-dir-mismatch']) == 0
    G.main(['m3'])
    G.main(['m4-ranks'])
    G.main(['tie-hosts'])
    assert all(seen[n].allow_run_dir_mismatch is False for n in ('cmd_m3', 'cmd_m4_ranks', 'cmd_tie_hosts'))
    with pytest.raises(SystemExit):                            # package keeps its own interface
        G.main(['package', '--allow-run-dir-mismatch'])


@pytest.mark.parametrize('content,reason', [
    ({'same_patient_pairs_masked': N_PAIRS, 'provenance': ['feature_dim', 12800]}, 'provenance is list'),
    ({'same_patient_pairs_masked': N_PAIRS, 'provenance': 'x'}, 'provenance is str'),
    ([1, 2], 'top level is list'),
    ('{not json', 'unreadable'),
    ({'same_patient_pairs_masked': [N_PAIRS], 'provenance': {'feature_dim': 12800}}, 'unhashable'),
])
def test_f8_malformed_rebuild_recorded_failure_and_failed(world, monkeypatch, content, reason):
    p = os.path.join(world['fd'], 'm1_rebuild_sampler.json')
    if isinstance(content, str):
        open(p, 'w').write(content)
    else:
        json.dump(content, open(p, 'w'))
    rep, _ = run(world)                                       # no bare exception
    assert not rep['pass']
    assert rep['first_failing_gate'].startswith('M1.2 rebuild sampler:') and reason in rep['first_failing_gate']
    assert rep['gates']['rebuild_malformed']['sampler']['path'] == p
    assert 'median_pooling' not in rep['gates']['rebuild_malformed']
    _patch_m3(world, monkeypatch)
    rd = world['rd']
    assert G.main(['m3', '--run-dir', rd, '--fidelity-dir', world['fd'], '--fp32', world['fp32'],
                   '--allow-run-dir-mismatch']) == 1
    out = json.load(open(os.path.join(rd, 'm3_matrix_gates.json')))
    assert not out['pass'] and reason in out['first_failing_gate']
    fl = json.load(open(os.path.join(rd, 'FAILED')))
    assert fl == {'first_failing_gate': out['first_failing_gate'],
                  'evidence_path': os.path.join(rd, 'm3_matrix_gates.json')}


def test_f8_wellformed_rebuild_has_no_malformed_record(world):
    rep, _ = run(world)
    assert rep['pass'] and 'rebuild_malformed' not in rep['gates']


# ----------------------------------------------------------------------------------------------
# ADDENDUM_003 (implementation_008): M4.1 reads config.input_similarity_matrix; real-shape fixture
# ----------------------------------------------------------------------------------------------
# Real-output-derived fixture (ADDENDUM_003 §2.6). Source (read-only):
#   prosper_revision_output/r2_1_virchow2_fp32_main_20261013/run_002/c2_metric_suite/
#   metrics_virchow2_fp32_ot_cosine.json, sha256 c69d570a9e0632373da413f9acc392cafaa5c11c5c13e611c2a8bb2928f368ea
#   (6669 bytes, written by the accepted prosper-search metrics_cli).
# Truncation rule: point_estimates and confidence_intervals cut to their first 2 entries in file order; config (all
# 15 keys), n_slides, neighbour_diagnostics and ranked_lists verbatim; key order preserved. Derivation script and
# provenance: implementation_008/fixture/{derive_fixture.py,FIXTURE_PROVENANCE.json} (derived sha256 cb243aa0...).
# Tests replace the real run_002 prefix REAL_RUN_002 with a synthetic stage dir and `ot_cosine` with each method.
REAL_RUN_002 = ('/PROSPER_WORKSPACE/prosper_revision/'
                'prosper_revision_output/r2_1_virchow2_fp32_main_20261013/run_002')
REAL_METRICS_OT_COSINE = json.loads(r'''
{
 "config": {
  "model": "virchow2_fp32",
  "method": "ot_cosine",
  "depth": 10,
  "k_values": [
   1,
   3,
   5,
   10
  ],
  "bootstrap_B": 5000,
  "seed": 123,
  "bootstrap_mode": "pps_cluster_one_slide",
  "dropped_labels": [
   "MISC"
  ],
  "tumor_only": false,
  "recall_denominator": "per-query: same class, different patient",
  "input_similarity_matrix": "/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output/r2_1_virchow2_fp32_main_20261013/run_002/similarity_matrix/combined_similarity_matrix_virchow2_fp32_ot_cosine_no_same_patient.pkl",
  "input_metadata": "/PROSPER_WORKSPACE/comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv",
  "same_patient_masked_input": true,
  "protocol": "configs/config.paper.yaml @ 1.0-frozen-20260827",
  "prosper_commit": "fd50373"
 },
 "n_slides": 2778,
 "neighbour_diagnostics": {
  "n_queries": 2778,
  "sentinels_encountered_before_depth": 0,
  "queries_with_fewer_than_depth_neighbours": 0
 },
 "point_estimates": {
  "n_queries": 2778,
  "retrieval_depth": 10
 },
 "confidence_intervals": {
  "macro_hit_at_1": {
   "point": 0.7239763813687694,
   "lo": 0.7022270517322063,
   "hi": 0.7455335057563776,
   "B_effective": 5000
  },
  "macro_precision_at_1": {
   "point": 0.7239763813687694,
   "lo": 0.7022270517322063,
   "hi": 0.7455335057563776,
   "B_effective": 5000
  }
 },
 "ranked_lists": "/PROSPER_WORKSPACE/prosper_revision/prosper_revision_output/r2_1_virchow2_fp32_main_20261013/run_002/c2_metric_suite/ranked_lists_virchow2_fp32_ot_cosine.csv.gz"
}
''')
REAL_CONFIG_KEYS = ['model', 'method', 'depth', 'k_values', 'bootstrap_B', 'seed', 'bootstrap_mode', 'dropped_labels',
                    'tumor_only', 'recall_denominator', 'input_similarity_matrix', 'input_metadata',
                    'same_patient_masked_input', 'protocol', 'prosper_commit']


def _real_metrics(stage_dir, m):
    """The real-shape metrics_cli document for method `m` with its paths in `stage_dir`."""
    txt = json.dumps(REAL_METRICS_OT_COSINE).replace(REAL_RUN_002, stage_dir).replace('ot_cosine', m)
    return json.loads(txt)


def _real_ranks_world(rd, stage_dir=None, edit=None):
    """Write real-shape metrics_virchow2_fp32_<m>.json (paths naming `stage_dir`, default rd) and ranks files."""
    os.makedirs(os.path.join(rd, 'c2_metric_suite'), exist_ok=True)
    for m in F.METHODS:
        r = os.path.join(rd, 'c2_metric_suite', f'ranked_lists_virchow2_fp32_{m}.csv.gz')
        if not os.path.exists(r):
            open(r, 'wb').write(b'x' + m.encode())
        d = _real_metrics(stage_dir or rd, m)
        if edit:
            edit(m, d)
        json.dump(d, open(os.path.join(rd, 'c2_metric_suite', f'metrics_virchow2_fp32_{m}.json'), 'w'))


def test_a8_real_fixture_shape_is_the_accepted_metrics_cli_shape():
    d = REAL_METRICS_OT_COSINE
    assert list(d) == ['config', 'n_slides', 'neighbour_diagnostics', 'point_estimates', 'confidence_intervals',
                       'ranked_lists']
    assert list(d['config']) == REAL_CONFIG_KEYS and 'input_similarity_matrix' not in d
    assert d['config']['input_similarity_matrix'] == os.path.join(
        REAL_RUN_002, 'similarity_matrix', 'combined_similarity_matrix_virchow2_fp32_ot_cosine_no_same_patient.pkl')
    assert d['config']['input_similarity_matrix'] == G.matrix_path(REAL_RUN_002, 'ot_cosine', True)
    assert d['ranked_lists'] == os.path.join(REAL_RUN_002, 'c2_metric_suite',
                                             'ranked_lists_virchow2_fp32_ot_cosine.csv.gz')
    assert G._metrics_matrix_field(d) == (d['config']['input_similarity_matrix'], 'config')


def test_a8_m4_real_shape_config_key_passes_and_records_config(world):
    """The run_002 defect: base gates read only the top level; the real (config-nested) shape now passes."""
    rd = world['rd']
    _real_ranks_world(rd)
    rep = G.m4_ranks_resolution(rd)
    assert rep['pass'], rep['failures']
    for m in F.METHODS:
        r = rep['methods'][m]
        assert r['input_similarity_matrix_key'] == 'config'
        assert r['input_similarity_matrix'] == G.matrix_path(rd, m, True)
        assert list(r['config']) == REAL_CONFIG_KEYS and r['config'] == _real_metrics(rd, m)['config']
        assert r['matrix_resolution'] == {'route': 'direct', 'ok': True}
        assert r['ranked_lists_resolution'] == {'route': 'direct', 'ok': True}
        assert r['metrics_json_sha256'] == F.sha256_file(r['metrics_json'])
        assert 'top_level_input_similarity_matrix_ignored' not in r


def test_a8_m4_top_level_fallback_and_both_absent(world):
    rd = world['rd']

    def top(m, d):                     # pre-ADDENDUM_003 synthetic shape: top level only
        d['input_similarity_matrix'] = d['config'].pop('input_similarity_matrix')
    _real_ranks_world(rd, edit=top)
    rep = G.m4_ranks_resolution(rd)
    assert rep['pass'], rep['failures']
    assert {r['input_similarity_matrix_key'] for r in rep['methods'].values()} == {'top_level'}

    def absent(m, d):
        if m == 'sampler':
            del d['config']['input_similarity_matrix']
        if m == 'yottixel':
            d['config']['input_similarity_matrix'] = ''        # empty string counts as absent
    _real_ranks_world(rd, edit=absent)
    rep = G.m4_ranks_resolution(rd)
    assert not rep['pass']
    for m in ('sampler', 'yottixel'):
        assert rep['methods'][m]['input_similarity_matrix_key'] is None
        assert rep['methods'][m]['input_similarity_matrix'] is None
        assert f'M4.1 {m}: resolved_is_fp32_run_matrix' in rep['failures']
        assert f'M4.1 {m}: not_published_tree' in rep['failures']
    assert rep['methods']['ot_cosine']['input_similarity_matrix_key'] == 'config'
    assert not any(f.startswith('M4.1 ot_cosine') or f.startswith('M4.1 median_pooling') for f in rep['failures'])
    os.remove(os.path.join(rd, 'c2_metric_suite', 'metrics_virchow2_fp32_median_pooling.json'))
    rep = G.m4_ranks_resolution(rd)
    r = rep['methods']['median_pooling']
    assert r['input_similarity_matrix_key'] is None and r['config'] is None and r['metrics_json_sha256'] is None
    assert 'M4.1 median_pooling: metrics_json_present' in rep['failures']


def test_a8_m4_config_wins_over_top_level(world):
    rd = world['rd']

    def both(m, d):
        if m == 'ot_cosine':           # config names the published tree, top level the run matrix -> fails
            d['input_similarity_matrix'] = d['config']['input_similarity_matrix']
            d['config']['input_similarity_matrix'] = os.path.join(F.PUBM, 'x.pkl')
        if m == 'sampler':             # config right, top level stale -> passes, stale value recorded
            d['input_similarity_matrix'] = '/elsewhere/old.pkl'
    _real_ranks_world(rd, edit=both)
    rep = G.m4_ranks_resolution(rd)
    o, s_ = rep['methods']['ot_cosine'], rep['methods']['sampler']
    assert o['input_similarity_matrix_key'] == 'config' and o['input_similarity_matrix'] == os.path.join(F.PUBM,
                                                                                                         'x.pkl')
    assert 'M4.1 ot_cosine: resolved_is_fp32_run_matrix' in rep['failures']
    assert 'M4.1 ot_cosine: not_published_tree' in rep['failures']
    assert o['top_level_input_similarity_matrix_ignored'] == G.matrix_path(rd, 'ot_cosine', True)
    assert s_['checks']['resolved_is_fp32_run_matrix'] is True
    assert s_['top_level_input_similarity_matrix_ignored'] == '/elsewhere/old.pkl'
    assert not any(f.startswith('M4.1 sampler') for f in rep['failures'])


def test_a8_m4_ranked_lists_field_checked(world):
    rd = world['rd']

    def rl(m, d):
        if m == 'yottixel':
            d['ranked_lists'] = os.path.join(rd, 'c2_metric_suite', 'other.csv.gz')
        if m == 'sampler':
            del d['ranked_lists']
    _real_ranks_world(rd, edit=rl)
    rep = G.m4_ranks_resolution(rd)
    assert rep['failures'] == ['M4.1 sampler: ranked_lists_field_matches', 'M4.1 yottixel: ranked_lists_field_matches']
    assert rep['methods']['sampler']['ranked_lists_resolution']['why'] == 'no ranked_lists field'


def test_a8_m4_records_script_hashes_and_step_receipts(world):
    rd = world['rd']
    _real_ranks_world(rd)
    jd = os.path.join(rd, 'jobs', 'm4_ranks_ot_cosine')
    os.makedirs(jd)
    open(os.path.join(jd, 'launch.sha256'), 'w').write(
        'a' * 64 + '  /x/prosper-search/metrics_cli.py\n' + 'b' * 64 + '  /x/launch.sh\nnot a hash line\n')
    json.dump({'stage': 'run_003', 'args': ['--method', 'ot_cosine'], 'module': 'metrics_cli', 'kind': 'cli'},
              open(os.path.join(jd, 'step.json'), 'w'))
    gd = os.path.join(rd, 'jobs', 'm4_ranks_gate')
    os.makedirs(gd)
    open(os.path.join(gd, 'launch.sha256'), 'w').write('c' * 64 + '  /x/gates.py\n')
    rep = G.m4_ranks_resolution(rd)
    assert rep['pass'], rep['failures']
    sh = rep['script_sha256']
    assert sh['gates']['sha256'] == F.sha256_file(os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_gates.py'))
    assert sh['fidelity']['sha256'] == F.sha256_file(os.path.join(REPO, 'scripts',
                                                                  'r2_1_virchow2_fp32_main_fidelity.py'))
    ps = rep['methods']['ot_cosine']['producing_step']
    assert ps['launch_hashes'] == {'/x/prosper-search/metrics_cli.py': 'a' * 64, '/x/launch.sh': 'b' * 64}
    assert ps['step_stage'] == 'run_003' and ps['step_module'] == 'metrics_cli'
    assert ps['launch.sha256']['sha256'] == F.sha256_file(os.path.join(jd, 'launch.sha256'))
    assert ps['step.json']['sha256'] == F.sha256_file(os.path.join(jd, 'step.json'))
    assert rep['methods']['sampler']['producing_step'] == {'step': 'm4_ranks_sampler', 'launch.sha256': None,
                                                          'step.json': None}
    assert rep['gate_step']['launch_hashes'] == {'/x/gates.py': 'c' * 64} and rep['gate_step']['step.json'] is None
    assert rep['stage_lineage'] is None


def _a8_lineage_world(world, tmp_path, monkeypatch, copy=False):
    """run_001 -> run_002 (init_run_002) -> run_002 metrics_cli outputs (real shape, run_002 paths) + similarity
    matrices -> FAILED -> run_003 (init_run_003). Returns (rd2, rd3)."""
    rd1, rd2, amend, sha = _a4_run002(world, tmp_path, monkeypatch)
    _real_ranks_world(rd2)
    rd3, add = _a8_run003(rd2, tmp_path, monkeypatch, copy=copy)
    return rd2, rd3


def test_a8_m4_carried_outputs_accepted_by_lineage_route(world, tmp_path, monkeypatch):
    """ADDENDUM_003 Issue A (ratification pending): carried metrics_cli outputs name run_002 paths; M4.1 accepts them
    in run_003 only as the lineage-recorded, hash-identical source copies."""
    rd2, rd3 = _a8_lineage_world(world, tmp_path, monkeypatch)
    for m in F.METHODS:
        d = json.load(open(os.path.join(rd3, 'c2_metric_suite', f'metrics_virchow2_fp32_{m}.json')))
        assert d['config']['input_similarity_matrix'] == G.matrix_path(rd2, m, True)
    assert G.main(['m4-ranks', '--run-dir', rd3]) == 0
    rep = json.load(open(os.path.join(rd3, 'm4_ranks_resolution.json')))
    assert rep['pass'] and rep['stage_lineage']['path'] == os.path.join(rd3, F.LINEAGE_RUN_003)
    for m in F.METHODS:
        r = rep['methods'][m]
        mr, rr = r['matrix_resolution'], r['ranked_lists_resolution']
        assert mr['route'] == rr['route'] == 'lineage' and mr['ok'] and rr['ok']
        assert mr['rel'] == f'similarity_matrix/combined_similarity_matrix_virchow2_fp32_{m}_no_same_patient.pkl'
        assert mr['sha256_expected'] == mr['sha256_recorded_path'] == mr['sha256_dest_recorded'] == \
            mr['sha256_source_recorded'] == r['checks']['matrix_sha256']
        assert mr['mode'] == 'hardlink' and mr['samefile_informational'] is True
        assert mr['lineage_schema'] == F.LINEAGE_RUN_003_SCHEMA


def test_a8_m4_lineage_route_copy_mode(world, tmp_path, monkeypatch):
    rd2, rd3 = _a8_lineage_world(world, tmp_path, monkeypatch, copy=True)
    rep = G.m4_ranks_resolution(rd3)
    assert rep['pass'], rep['failures']
    mr = rep['methods']['yottixel']['matrix_resolution']
    assert mr['mode'] == 'copy' and mr['samefile_informational'] is False


def test_a8_m4_lineage_route_negatives(world, tmp_path, monkeypatch):
    rd2, rd3 = _a8_lineage_world(world, tmp_path, monkeypatch, copy=True)
    # (1) sha mismatch: the run_003 copy differs from the lineage record
    open(G.matrix_path(rd3, 'sampler', True), 'ab').write(b'tamper')
    rep = G.m4_ranks_resolution(rd3)
    assert rep['failures'] == ['M4.1 sampler: resolved_is_fp32_run_matrix']
    assert rep['methods']['sampler']['matrix_resolution']['why'] == 'sha256 differs from the lineage record'
    # (2) recorded path is not the source copy of the expected path
    mj = os.path.join(rd3, 'c2_metric_suite', 'metrics_virchow2_fp32_yottixel.json')
    os.chmod(mj, 0o644)
    d = json.load(open(mj))
    d['config']['input_similarity_matrix'] = G.matrix_path(rd2, 'ot_cosine', True)
    os.remove(mj)
    json.dump(d, open(mj, 'w'))
    rep = G.m4_ranks_resolution(rd3)
    assert 'M4.1 yottixel: resolved_is_fp32_run_matrix' in rep['failures']
    assert rep['methods']['yottixel']['matrix_resolution']['why'] == \
        'recorded path is not the lineage source copy of the expected path'
    # (3) no own lineage record: the carried LINEAGE_RUN_002.json (dest_dir run_002) is never used
    os.rename(os.path.join(rd3, F.LINEAGE_RUN_003), os.path.join(str(tmp_path), 'moved_lineage.json'))
    assert G.stage_lineage(rd3) == (None, None) and os.path.exists(os.path.join(rd3, F.LINEAGE_RUN_002))
    rep = G.m4_ranks_resolution(rd3)
    assert not rep['pass'] and rep['stage_lineage'] is None
    for m in ('median_pooling', 'ot_cosine'):
        assert rep['methods'][m]['matrix_resolution']['why'] == 'no own run-lineage record'
        assert f'M4.1 {m}: resolved_is_fp32_run_matrix' in rep['failures']
        assert f'M4.1 {m}: ranked_lists_field_matches' in rep['failures']
    # (4) a lineage record whose dest_dir is another dir is not this stage's record
    lin = json.load(open(os.path.join(str(tmp_path), 'moved_lineage.json')))
    lin['dest_dir'] = rd2
    json.dump(lin, open(os.path.join(rd3, F.LINEAGE_RUN_003), 'w'))
    assert G.stage_lineage(rd3) == (None, None)


def test_a8_m4_lineage_route_refuses_append_only_and_outside(world, tmp_path, monkeypatch):
    rd2, rd3 = _a8_lineage_world(world, tmp_path, monkeypatch)
    lin = G.stage_lineage(rd3)
    assert lin[0] == os.path.join(rd3, F.LINEAGE_RUN_003)
    ok, rec = G.lineage_equivalent(rd3, os.path.join(rd3, 'journal.jsonl'), os.path.join(rd2, 'journal.jsonl'), lin)
    assert not ok and rec['why'] == 'expected path not a (single, non-append-only) lineage-listed file'
    ok, rec = G.lineage_equivalent(rd3, os.path.join(str(tmp_path), 'x.pkl'), os.path.join(rd2, 'x.pkl'), lin)
    assert not ok and rec['why'] == 'expected path outside the run dir'
    ok, rec = G.lineage_equivalent(rd3, G.matrix_path(rd3, 'sampler', True), None, lin)
    assert not ok and rec['why'] == 'no path recorded'
