"""Synthetic-fixture tests for scripts/r2_1_virchow2_fp32_main_fidelity.py (V2-main M1.0-M1.5, M2).

Tiny synthetic banks/matrices/logs only; no real bank, matrix or LSF interaction. Run with an explicit
--basetemp outside /tmp, `-p no:cacheprovider` and PYTHONDONTWRITEBYTECODE=1.
"""
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('v2main_fidelity_under_test',
                                               os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_fidelity.py'))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

LSF_OK = """[worker] host=noderome1 job=1 idx=1
{body}
Sender: LSF System <lsfadmin@noderome1>
Subject: Job 555[3]: <x[3]> in cluster <c> Done

Job <x[3]> was submitted from host <h> by user <u> in cluster <c> at Sun Oct 11 09:07:47 2026
Job was executed on host(s) <noderome189>, in queue <heavy_io>, as user <u> in cluster <c> at Sun Oct 11 09:07:57 2026
------------------------------------------------------------

{status}

Resource usage summary:

    CPU time :                                   36.10 sec.
    Max Memory :                                 1068 MB
    Run time :                                   100 sec.
"""


def lsf_text(body, status='Successfully completed.'):
    return LSF_OK.format(body=body, status=status)


def cli_block(obj, verdict='PASS'):
    return json.dumps(obj, indent=2) + f'\nFIDELITY: {verdict}'


# ----------------------------------------------------------------------------------------------
# LSF parsing
# ----------------------------------------------------------------------------------------------

def test_parse_lsf_out_success_and_failure():
    d = F.parse_lsf_out(lsf_text('hello'))
    assert d['rc'] == 0 and d['run_time_s'] == 100 and d['max_memory_mb'] == 1068.0
    assert d['host'] == 'noderome189' and d['queue'] == 'heavy_io' and d['job_id'] == '555' and d['job_index'] == '3'
    d = F.parse_lsf_out(lsf_text('x', 'TERM_MEMLIMIT: job killed after reaching LSF memory usage limit.\n'
                                      'Exited with exit code 130.'))
    assert d['rc'] == 130 and d['term_reason'] == 'TERM_MEMLIMIT'
    assert F.parse_lsf_out('still running')['rc'] is None
    assert F.program_stdout(lsf_text('ABC')).strip().endswith('ABC')


def test_parse_cli_json_and_verdict():
    obj, v = F.parse_cli_json_and_verdict(lsf_text('[aggregate] note\n' + cli_block({'a': 1, 'b': {'c': 2}})))
    assert obj == {'a': 1, 'b': {'c': 2}} and v == 'PASS'
    with pytest.raises(F.FidelityError):
        F.parse_cli_json_and_verdict(lsf_text(cli_block({'a': 1}) + '\nFIDELITY: PASS'))
    with pytest.raises(F.FidelityError):
        F.parse_cli_json_and_verdict(lsf_text('no verdict'))


# ----------------------------------------------------------------------------------------------
# M1.3 probe derivation
# ----------------------------------------------------------------------------------------------

def _inv(n=20, seed=0):
    rng = np.random.default_rng(seed)
    tiles = rng.integers(5, 60, size=n)
    tiles[3] = tiles[7] = 5            # tie on the smallest count resolved by str(slide_id)
    ids = [str(x) for x in rng.choice(np.arange(1000, 30000), size=n, replace=False)]
    dt = ['float32' if i % 4 == 0 else 'float16' for i in range(n)]
    return [{'bank': 'virchow2', 'slide_id': s, 'canonical_tiles': int(t), 'dtype': d, 'size': 1, 'sha256': ''}
            for s, t, d in zip(ids, tiles, dt)]


def _independent_probe_table(inv):
    n = len(inv)
    srt = sorted(inv, key=lambda r: (r['canonical_tiles'], r['slide_id']))
    sel = [srt[i] for i in (0, 1, (n - 1) // 2, (n - 1) // 2 + 1, n - 2, n - 1)]
    f16 = sorted([r for r in inv if r['dtype'] == 'float16'], key=lambda r: r['canonical_tiles'])
    if f16[-1]['slide_id'] not in {r['slide_id'] for r in sel}:
        sel.append(f16[-1])
    ids = sorted(r['slide_id'] for r in inv)
    return tuple((r['slide_id'], r['canonical_tiles'], r['dtype'], ids.index(r['slide_id']) + 1) for r in sel)


def test_derive_probes_matches_independent_rule(monkeypatch):
    inv = _inv()
    monkeypatch.setattr(F, 'N_SLIDES', len(inv))
    ids = sorted(r['slide_id'] for r in inv)
    table = _independent_probe_table(inv)
    rep = F.derive_probes(inv, ids, ids, table=table)
    assert rep['equal_to_contract_table'] and len(rep['derived']) == len(table)
    bad = list(table)
    bad[2] = (bad[2][0], bad[2][1], bad[2][2], bad[2][3] + 1)
    with pytest.raises(F.FidelityError, match='M1.3'):
        F.derive_probes(inv, ids, ids, table=tuple(bad))
    with pytest.raises(F.FidelityError, match='TMPF glob'):
        F.derive_probes(inv, ids, ids[:-1], table=table)


def test_contract_probe_table_constants():
    """Self-consistency of the specification table as transcribed: 3 float32 + 4 float16, tiles 9..5051."""
    assert sum(d == 'float32' for _, _, d, _ in F.PROBE_TABLE) == 3
    assert [t for _, t, _, _ in F.PROBE_TABLE] == [9, 9, 1299, 1300, 4829, 5051, 4437]
    assert {i for *_, i in F.PROBE_TABLE} == {911, 2431, 1682, 2585, 497, 1349, 509}


# ----------------------------------------------------------------------------------------------
# M1.0 + M2 bank check
# ----------------------------------------------------------------------------------------------

W = 8


def make_banks(root, n16=4, n32=2, seed=3):
    rng = np.random.default_rng(seed)
    tm, fp = os.path.join(root, 'tmpf'), os.path.join(root, 'fp32')
    os.makedirs(tm)
    os.makedirs(fp)
    inv, slides, ref_meds = [], {}, []
    for i in range(n16 + n32):
        sid = str(100 + i * 7)
        n = 3 + i
        x = rng.normal(0, 1, size=(n, W)).astype(np.float32)
        coords = [{'wsi_loc': [int(a), int(b)]} for a, b in rng.integers(0, 9999, size=(n, 2))]
        if i < n16:
            stored = x.astype(np.float16)
            new = (stored.astype(np.float32) + rng.normal(0, 1e-3, size=(n, W)).astype(np.float32)).astype(np.float32)
            n64, o64 = new.astype(np.float16).astype(np.float64), stored.astype(np.float64)
            ref_meds.append(float(np.median(np.linalg.norm(n64 - o64, axis=1) / np.linalg.norm(o64, axis=1))))
        else:
            stored = x
            new = x.copy()
        bt = pickle.dumps((stored, coords))
        bf = pickle.dumps((np.ascontiguousarray(new), [dict(c) for c in coords]))
        open(os.path.join(tm, sid + '.pkl'), 'wb').write(bt)
        open(os.path.join(fp, sid + '.pkl'), 'wb').write(bf)
        open(os.path.join(fp, sid + '.record.json'), 'w').write('{}')
        inv.append({'bank': 'virchow2', 'slide_id': sid, 'canonical_tiles': n, 'dtype': str(stored.dtype),
                    'size': len(bt), 'sha256': hashlib.sha256(bt).hexdigest()})
        slides[sid] = {'pickle_sha256': hashlib.sha256(bf).hexdigest()}
    ref = float(np.median(ref_meds))
    return {'tmpf': tm, 'fp32': fp, 'inv': inv,
            'reconcile': {'disposition': 'COMPLETE_PASS', 'slides': slides},
            'af8': {'PASS': True, 'n_checked': n16 + n32, 'coords_mismatch_slides': []},
            'fid': {'fp16': {'median_rel_l2_distribution': {'0.5': ref}}}}


@pytest.fixture
def small(monkeypatch, tmp_path):
    monkeypatch.setattr(F, 'N_SLIDES', 6)
    monkeypatch.setattr(F, 'N_FLOAT16', 4)
    monkeypatch.setattr(F, 'N_FLOAT32', 2)
    monkeypatch.setattr(F, 'WIDTH', W)
    return make_banks(str(tmp_path))


def run_bank(b):
    return F.bank_check(b['inv'], b['reconcile'], b['af8'], b['fid'], b['tmpf'], b['fp32'])


def test_bank_check_pass_and_m7(small):
    m1, m2, fails = run_bank(small)
    assert fails == [] and m1['pass'] and m2['pass']
    assert m1['dtype_counts_observed'] == {'float16': 4, 'float32': 2}
    m7 = m2['m7_reproduction']
    assert m7['n_slides'] == 4 and m7['reproduces_to_3_significant_figures']
    assert m7['value'] == small['fid']['fp16']['median_rel_l2_distribution']['0.5']
    assert m2['af8_mosaic_check_pass_all_2921'] and m2['record_json_siblings'] == 6


def _rewrite(path, obj):
    open(path, 'wb').write(pickle.dumps(obj))


@pytest.mark.parametrize('case,match', [
    ('tmpf_sha', 'M1.0'), ('fp32_sha', 'sha256'), ('coords_order', 'coordinates'), ('float64', 'dtype float64'),
    ('fortran', 'C-contiguous'), ('nan', 'non-finite'), ('rows', 'canonical_tiles'), ('width', 'width'),
    ('missing_fp32', 'M2.1'), ('disposition', 'disposition'), ('tuple3', '2-tuple')])
def test_bank_check_negative(small, case, match):
    b = small
    sid = b['inv'][1]['slide_id']
    pf = os.path.join(b['fp32'], sid + '.pkl')
    a, c = pickle.load(open(pf, 'rb'))
    if case == 'tmpf_sha':
        b['inv'][1]['sha256'] = '0' * 64
    elif case == 'fp32_sha':
        b['reconcile']['slides'][sid]['pickle_sha256'] = '0' * 64
    elif case == 'coords_order':
        _rewrite(pf, (a, c[::-1]))
    elif case == 'float64':
        _rewrite(pf, (a.astype(np.float64), c))
    elif case == 'fortran':
        _rewrite(pf, (np.asfortranarray(a), c))
    elif case == 'nan':
        a = a.copy()
        a[0, 0] = np.nan
        _rewrite(pf, (a, c))
    elif case == 'rows':
        _rewrite(pf, (a[:-1].copy(), c[:-1]))
    elif case == 'width':
        _rewrite(pf, (np.ascontiguousarray(a[:, :-1]), c))
    elif case == 'missing_fp32':
        os.remove(pf)
    elif case == 'disposition':
        b['reconcile']['disposition'] = 'PARTIAL'
    elif case == 'tuple3':
        _rewrite(pf, (a, c, None))
    if case not in ('fp32_sha', 'tmpf_sha', 'missing_fp32', 'disposition'):
        b['reconcile']['slides'][sid]['pickle_sha256'] = hashlib.sha256(open(pf, 'rb').read()).hexdigest()
    _, _, fails = run_bank(b)
    assert fails and any(match in f for f in fails), fails


def test_m7_mismatch_is_reported_not_stop(small):
    small['fid']['fp16']['median_rel_l2_distribution']['0.5'] *= 1.5
    m1, m2, fails = run_bank(small)
    assert fails == [] and m2['pass'] and not m2['m7_reproduction']['reproduces_to_3_significant_figures']


# ----------------------------------------------------------------------------------------------
# M1.1 aggregate gate
# ----------------------------------------------------------------------------------------------

def agg_obj(**kw):
    o = {'encoder': 'virchow2', 'method': 'sampler', 'metric': 'cityblock', 'n_common_slides': 2921,
         'n_compared_cells': 2921 * 2920, 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0, 'max_rel_diff': 0.0}
    o.update(kw)
    return o


def test_aggregate_gate_pass():
    r = F.aggregate_gate('sampler', lsf_text(cli_block(agg_obj())))
    assert r['pass'] and r['q5_exactness']['max_abs_diff_exactly_0']


@pytest.mark.parametrize('kw,verdict,status,key', [
    ({'max_rel_diff': 1e-6}, 'PASS', 'Successfully completed.', 'max_rel_diff_lt_1e-6'),
    ({}, 'FAIL', 'Successfully completed.', 'fidelity_line_pass'),
    ({}, 'PASS', 'Exited with exit code 1.', 'rc_0'),
    ({'n_common_slides': 2920}, 'PASS', 'Successfully completed.', 'n_common_slides_2921'),
    ({'n_compared_cells': 2921 * 2920 - 1}, 'PASS', 'Successfully completed.', 'n_compared_cells_8529320'),
    ({'method': 'median_pooling'}, 'PASS', 'Successfully completed.', 'encoder_method')])
def test_aggregate_gate_fail(kw, verdict, status, key):
    r = F.aggregate_gate('sampler', lsf_text(cli_block(agg_obj(**kw), verdict), status))
    assert not r['pass'] and not r['checks'][key]


# ----------------------------------------------------------------------------------------------
# M1.3 probe gate + stderr census
# ----------------------------------------------------------------------------------------------

def _probe_obj(**kw):
    o = {'encoder': 'virchow2', 'method': 'yottixel', 'n_probe_rows': 7, 'n_common_cols': 2921,
         'n_compared_cells': 7 * 2920, 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0, 'max_rel_diff': 0.0}
    o.update(kw)
    return o


def _write_logs(stage, job, idx, err='', out='[worker] ok\n'):
    d = os.path.join(stage, 'lsf')
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f'{job}.900{idx}_{idx}.err'), 'w').write(err)
    open(os.path.join(d, f'{job}.900{idx}_{idx}.out'), 'w').write(lsf_text(out))


def test_stderr_census_and_probe_gate(tmp_path):
    stage = str(tmp_path)
    idx = [911, 2431, 1682, 2585, 497, 1349, 509]
    for i in idx:
        _write_logs(stage, 'pbob', i, err='UserWarning: x\n' if i == 497 else '')
    _write_logs(stage, 'pbob', 1349, err='ot.lp numItermax reached\nRuntimeWarning: y\n')
    prov = os.path.join(stage, 'logs', 'similarity_virchow2_bob_probe')
    os.makedirs(prov)
    for i in idx:
        open(os.path.join(prov, f'provenance_{i}.json'), 'w').write('{}')
    rows = F.stderr_census(stage, ['pbob'])
    assert len(rows) == 7 and sum(r['UserWarning'] for r in rows) == 1
    assert [r['numItermax'] for r in rows if r['index'] == 1349] == [1]
    ok = F.probe_gate('bob', lsf_text(cli_block(_probe_obj())), prov, idx, [], rows, [])
    assert ok['pass'] and ok['numItermax_elements']        # numItermax reported, bar passes -> not a STOP
    bad = F.probe_gate('bob', lsf_text(cli_block(_probe_obj(max_rel_diff=2e-6), 'FAIL')), prov, idx, [], rows, [])
    assert not bad['checks']['no_numItermax_with_failed_bar']
    os.remove(os.path.join(prov, 'provenance_509.json'))
    open(os.path.join(prov, 'provenance_1_7.json'), 'w').write('{}')
    r = F.probe_gate('bob', lsf_text(cli_block(_probe_obj())), prov, idx, [], rows, [])
    assert not r['checks']['provenance_exactly_7']
    r = F.probe_gate('bob', lsf_text(cli_block(_probe_obj(n_compared_cells=20439))), prov, idx,
                     [{'line': '[similarity] a vs b: FAILED'}], rows, [])
    assert not r['checks']['n_compared_cells_20440'] and not r['checks']['no_FAILED_stdout_line']


def test_census_traceback_and_failed_stdout(tmp_path):
    stage = str(tmp_path)
    _write_logs(stage, 'pot', 5, err='Traceback (most recent call last):\n', out='[similarity] 1 vs 2: FAILED x\n')
    rows = F.stderr_census(stage, ['pot'])
    g = F.census_gate(rows)
    assert not g['pass'] and g['traceback_elements']
    assert F.failed_stdout_lines(stage, 'pot')[0]['index'] == 5
    p = os.path.join(stage, 'stderr_census.tsv')
    F.write_census_tsv(p, rows)
    head, line = open(p).read().splitlines()
    assert head.split('\t')[3:8] == list(F.CENSUS_KEYS) and line.split('\t')[6] == '1'
    with pytest.raises(F.FidelityError):
        F.write_census_tsv(p, rows)
    assert not F.census_gate([])['pass']


def test_latest_attempts_prefers_retry(tmp_path):
    d = os.path.join(str(tmp_path), 'lsf')
    os.makedirs(d)
    for jid in (100, 250):
        open(os.path.join(d, f'job.{jid}_7.out'), 'w').write('x')
    open(os.path.join(d, 'jobX.300_7.out'), 'w').write('x')     # other job name, not matched
    assert F.latest_attempts(str(tmp_path), 'job') == {7: (250, os.path.join(d, 'job.250_7.out'))}


def test_per_probe_rows(tmp_path):
    ids = ['3', '10', '22', '4']
    rng = np.random.default_rng(1)
    M = np.abs(rng.normal(size=(4, 4)))
    np.fill_diagonal(M, 1e6)
    ref = pd.DataFrame(M, index=[int(i) for i in ids], columns=[int(i) for i in ids])
    sd = os.path.join(str(tmp_path), 'sim')
    os.makedirs(sd)
    for qi in (0, 2):
        row = [(ids[j], float(M[qi, j]) * (1 + (1e-7 if qi == 2 and j == 1 else 0))) for j in range(4)]
        pickle.dump(row, open(os.path.join(sd, f'similarity_{ids[qi]}.pkl'), 'wb'))
    out = {r['slide_id']: r for r in F.per_probe_rows(sd, ref)}
    assert out['3']['bit_identical_all_cols'] and out['3']['max_abs'] == 0.0 and out['3']['n_live'] == 3
    assert not out['22']['bit_identical_all_cols'] and 0 < out['22']['max_rel'] < 2e-7
    assert out['22']['sentinel_pattern_equal']


# ----------------------------------------------------------------------------------------------
# M1.2 rebuild gate
# ----------------------------------------------------------------------------------------------

def _mats(tmp, perturb=0.0, extra_sentinel=False, int_ids=False):
    rng = np.random.default_rng(5)
    ids = ['11', '7', '30', '2', '19']
    X = rng.normal(size=(5, 3))
    D = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    U = D.copy()
    np.fill_diagonal(U, 1e6)
    Mk = U.copy()
    Mk[0, 1] = Mk[1, 0] = 1e6
    pub_u = pd.DataFrame(U, index=ids, columns=ids)
    pub_m = pd.DataFrame(Mk, index=ids, columns=ids)
    U2, M2 = U.copy(), Mk.copy()
    U2[2, 3] += perturb
    M2[2, 3] += perturb
    if extra_sentinel:
        M2[3, 4] = M2[4, 3] = 1e6
    rid = [int(i) for i in ids] if int_ids else ids
    paths = {}
    for name, df in (('pu', pub_u), ('pm', pub_m), ('ru', pd.DataFrame(U2, index=rid, columns=rid)),
                     ('rm', pd.DataFrame(M2, index=rid, columns=rid))):
        paths[name] = os.path.join(tmp, name + '.pkl')
        pickle.dump(df, open(paths[name], 'wb'))
    return paths


def test_rebuild_gate_exact_and_bytes(tmp_path):
    p = _mats(str(tmp_path))
    r = F.rebuild_gate('sampler', p['ru'], p['rm'], p['pu'], p['pm'], {'same_patient_pairs_masked': 1})
    assert r['pass'] and r['q5_exactness']['masked_byte_identical'] and r['same_patient_pairs_masked'] == 1


def test_rebuild_gate_str_ids_tiny_diff_reported(tmp_path):
    p = _mats(str(tmp_path), perturb=1e-12, int_ids=True)
    r = F.rebuild_gate('sampler', p['ru'], p['rm'], p['pu'], p['pm'], {})
    assert r['pass'] and not r['q5_exactness']['masked_byte_identical']
    assert r['q5_exactness']['detail']['masked']['n_differing_cells'] == 1


def test_rebuild_gate_pattern_and_bar_fail(tmp_path):
    p = _mats(str(tmp_path), extra_sentinel=True)
    r = F.rebuild_gate('sampler', p['ru'], p['rm'], p['pu'], p['pm'], {})
    assert not r['pass'] and not r['checks']['masked_pattern_equal']
    p = _mats(os.path.join(str(tmp_path)), perturb=1.0)
    r = F.rebuild_gate('sampler', p['ru'], p['rm'], p['pu'], p['pm'], {})
    assert not r['checks']['masked_live_max_rel_lt_1e-6']


# ----------------------------------------------------------------------------------------------
# M1.4 ranks
# ----------------------------------------------------------------------------------------------

def test_ranks_gate(tmp_path):
    raw = b'query,rank\n1,1\n'
    ref = os.path.join(str(tmp_path), 'ref.csv.gz')
    open(ref, 'wb').write(gzip.compress(raw, mtime=1))
    regen_dir = os.path.join(str(tmp_path), 'c2')
    os.makedirs(regen_dir)
    reg = os.path.join(regen_dir, 'ranked_lists_virchow2_sampler.csv.gz')
    open(reg, 'wb').write(gzip.compress(raw, mtime=999))
    sha = hashlib.sha256(open(ref, 'rb').read()).hexdigest()
    r = F.ranks_gate('sampler', regen_dir, ref, sha)
    assert r['pass'] and not r['gzip_file_bytes_equal']
    os.remove(reg)
    open(reg, 'wb').write(gzip.compress(raw + b'2,1\n'))
    assert not F.ranks_gate('sampler', regen_dir, ref, sha)['pass']
    open(os.path.join(regen_dir, 'ranked_lists_virchow2_sampler.parquet'), 'wb').write(b'')
    assert not F.ranks_gate('sampler', regen_dir, ref, sha)['checks']['no_parquet_written']
    with pytest.raises(F.FidelityError, match='M0.3'):
        F.ranks_gate('sampler', regen_dir, ref, '0' * 64)


def test_ranks_precheck(tmp_path):
    assert F.ranks_precheck(str(tmp_path))['pass']
    d = os.path.join(str(tmp_path), 'similarity_matrix')
    os.makedirs(d)
    open(os.path.join(d, 'combined_similarity_matrix_virchow2_pubrebuild_sampler.pkl'), 'w').write('')
    assert F.ranks_precheck(str(tmp_path))['pass']          # the pubrebuild token does not shadow PUBM
    open(os.path.join(d, 'combined_similarity_matrix_virchow2_sampler_no_same_patient.pkl'), 'w').write('')
    with pytest.raises(F.FidelityError, match='shadow'):
        F.ranks_precheck(str(tmp_path))


# ----------------------------------------------------------------------------------------------
# M1.5 report / markers
# ----------------------------------------------------------------------------------------------

def _gates(stage, fail=None, skip=None):
    for g in F.GATE_ORDER:
        if g == skip:
            continue
        ok = g != fail
        json.dump({'stage': g, 'pass': ok, 'failures': [] if ok else [f'{g} broke']},
                  open(os.path.join(stage, f'{g}.json'), 'w'))


def test_report_complete_failed_missing(tmp_path):
    for case in ('ok', 'fail', 'missing'):
        st = os.path.join(str(tmp_path), case)
        os.makedirs(st)
        _gates(st, fail='m1_ot_probe' if case == 'fail' else None, skip='m1_ranks_sampler' if case == 'missing'
               else None)
        rep = F.assemble_report(st)
        name = F.write_marker(st, rep)
        if case == 'ok':
            assert name == 'COMPLETE' and rep['all_pass'] and not os.path.exists(os.path.join(st, 'FAILED'))
        elif case == 'fail':
            assert name == 'FAILED' and rep['first_failing_gate'].startswith('m1_ot_probe')
        else:
            assert name == 'FAILED' and rep['first_failing_gate'] == 'm1_ranks_sampler: missing'
        with pytest.raises(F.FidelityError):
            F.write_marker(st, rep)


def test_report_failed_elements_forces_failed(tmp_path):
    st = str(tmp_path)
    _gates(st)
    open(os.path.join(st, 'failed_elements.jsonl'), 'w').write(json.dumps({'step': 'm1_ot_probe', 'index': 911}) + '\n')
    rep = F.assemble_report(st)
    assert not rep['all_pass'] and rep['first_failing_gate'].startswith('failed_elements.jsonl')
    assert F.write_marker(st, rep) == 'FAILED'


def test_write_json_new_never_overwrites(tmp_path):
    p = os.path.join(str(tmp_path), 'a.json')
    F.write_json_new(p, {'x': np.int64(1), 'y': np.float32(0.5)})
    assert json.load(open(p)) == {'x': 1, 'y': 0.5}
    with pytest.raises(F.FidelityError):
        F.write_json_new(p, {})


# ---------------------------------------------------------------------------------------------- repair 002 (F7)

def test_f7_rebuild_gate_rejects_nan(tmp_path):
    """Review probe P6: 3 NaNs in a rebuilt masked matrix must fail M1.2 (n_live 9 vs 12 published)."""
    p = _mats(str(tmp_path))
    df = pickle.load(open(p['rm'], 'rb'))
    a = df.to_numpy().copy()
    a[2, 3] = a[3, 2] = a[4, 0] = np.nan
    pickle.dump(pd.DataFrame(a, index=df.index, columns=df.columns), open(p['rm'], 'wb'))
    r = F.rebuild_gate('sampler', p['ru'], p['rm'], p['pu'], p['pm'], {})
    assert not r['pass'] and not r['checks']['masked_no_nan'] and not r['checks']['masked_n_live_eq_published']
    assert r['checks']['masked_pattern_equal'] and r['checks']['masked_live_max_rel_lt_1e-6']   # blind before F7
    assert r['masked_vs_published']['n_nan_mine'] == 3
    assert r['masked_vs_published']['n_live'] + 3 == r['masked_vs_published']['n_live_published']
    os.makedirs(os.path.join(str(tmp_path), 'u'))
    p2 = _mats(os.path.join(str(tmp_path), 'u'))
    du = pickle.load(open(p2['ru'], 'rb'))
    u = du.to_numpy().copy()
    u[1, 2] = np.nan
    pickle.dump(pd.DataFrame(u, index=du.index, columns=du.columns), open(p2['ru'], 'wb'))
    r = F.rebuild_gate('sampler', p2['ru'], p2['rm'], p2['pu'], p2['pm'], {})
    assert not r['pass'] and not r['checks']['unmasked_no_nan'] and r['checks']['masked_no_nan']


def test_f7_clean_rebuild_reports_counts(tmp_path):
    p = _mats(str(tmp_path))
    r = F.rebuild_gate('sampler', p['ru'], p['rm'], p['pu'], p['pm'], {})
    assert r['pass'] and r['checks']['masked_no_nan'] and r['checks']['masked_n_live_eq_published']
    assert r['masked_vs_published']['n_live'] == r['masked_vs_published']['n_live_published'] == 18


def _probe_rows_with(tmp_path, nan_cols=(), drop_cols=()):
    ids = ['3', '10', '22', '4']
    rng = np.random.default_rng(1)
    M = np.abs(rng.normal(size=(4, 4)))
    np.fill_diagonal(M, 1e6)
    ref = pd.DataFrame(M, index=[int(i) for i in ids], columns=[int(i) for i in ids])
    sd = os.path.join(str(tmp_path), 'sim')
    os.makedirs(sd)
    row = [(ids[j], float('nan') if j in nan_cols else float(M[0, j])) for j in range(4) if j not in drop_cols]
    pickle.dump(row, open(os.path.join(sd, 'similarity_3.pkl'), 'wb'))
    return F.per_probe_rows(sd, ref)


def test_f7_per_probe_row_nan_and_live_count(tmp_path):
    ok = _probe_rows_with(tmp_path / 'a')
    assert ok[0]['no_nan'] and ok[0]['n_live_eq_published'] and ok[0]['n_live_published'] == 3
    bad = _probe_rows_with(tmp_path / 'b', nan_cols=(1, 2))
    assert not bad[0]['no_nan'] and bad[0]['n_nan'] == 2 and not bad[0]['n_live_eq_published']
    short = _probe_rows_with(tmp_path / 'c', drop_cols=(2,))
    assert short[0]['no_nan'] and not short[0]['n_live_eq_published']
    stage = str(tmp_path / 'st')
    idx = [911, 2431, 1682, 2585, 497, 1349, 509]
    for i in idx:
        _write_logs(stage, 'pbob', i)
    prov = os.path.join(stage, 'logs', 'similarity_virchow2_bob_probe')
    os.makedirs(prov)
    for i in idx:
        open(os.path.join(prov, f'provenance_{i}.json'), 'w').write('{}')
    rows = F.stderr_census(stage, ['pbob'])
    txt = lsf_text(cli_block(_probe_obj()))
    assert F.probe_gate('bob', txt, prov, idx, [], rows, ok)['pass']
    g = F.probe_gate('bob', txt, prov, idx, [], rows, bad)
    assert not g['pass'] and not g['checks']['per_row_no_nan'] and not g['checks']['per_row_n_live_eq_published']
    g = F.probe_gate('bob', txt, prov, idx, [], rows, short)
    assert not g['pass'] and g['checks']['per_row_no_nan'] and not g['checks']['per_row_n_live_eq_published']


# ----------------------------------------------------------------------------------------------
# AMENDMENT_003: amended M1.3 OT gate, init-002 lineage, default stage directory
# ----------------------------------------------------------------------------------------------

A3_IDX = [911, 2431, 1682, 2585, 497, 1349, 509]
A3_AMEND = {'path': '/x/contract_amendment_003/AMENDMENT_003.md', 'sha256': 'a' * 64}


def _a3_rows(tmp_path, scale=7.4e-6, tie_swap_rank=None, n=70):
    """7 probe rows of a synthetic published OT matrix; every recomputed live cell = published * (1 + scale).
    tie_swap_rank r: in the first probe row the published values at sorted ranks r, r+1 are made near-tied
    (1e-9 apart) and the recomputed row swaps them (rel diff ~1e-9, so only the ordering changes)."""
    rng = np.random.default_rng(11)
    ids = [str(1000 + i) for i in range(n)]
    M = rng.uniform(0.2, 1.0, size=(n, n))
    np.fill_diagonal(M, 1e6)
    R = np.where(M < 1e6, M * (1 + scale), M)
    if tie_swap_rank is not None:
        order = np.argsort(np.where(M[0] < 1e6, M[0], np.inf), kind='stable')
        j1, j2 = order[tie_swap_rank], order[tie_swap_rank + 1]
        M[0, j2] = M[0, j1] + 1e-9
        R[0, j1], R[0, j2] = M[0, j2], M[0, j1]
    ref = pd.DataFrame(M, index=[int(i) for i in ids], columns=[int(i) for i in ids])
    sd = os.path.join(str(tmp_path), 'sim')
    os.makedirs(sd)
    for qi in range(7):
        pickle.dump([(ids[j], float(R[qi, j])) for j in range(n)], open(os.path.join(sd, f'similarity_{ids[qi]}.pkl'),
                                                                        'wb'))
    return F.per_probe_rows(sd, ref, rank_depths=(10, 50)), ref, sd


def _a3_stage(tmp_path, nit=False):
    stage = os.path.join(str(tmp_path), 'stage')
    for i in A3_IDX:
        _write_logs(stage, 'pot', i, err='ot.lp numItermax reached\n' if nit and i == 509 else '')
    prov = os.path.join(stage, 'logs', 'similarity_virchow2_ot_probe')
    os.makedirs(prov)
    for i in A3_IDX:
        open(os.path.join(prov, f'provenance_{i}.json'), 'w').write('{}')
    return stage, prov, F.stderr_census(stage, ['pot'])


def _ot_cli(max_rel=7.416963761463222e-06, verdict='FAIL', status='Exited with exit code 1.'):
    return lsf_text(cli_block(_probe_obj(method='ot_cosine', max_rel_diff=max_rel, max_abs_diff=6.03e-6), verdict),
                    status)


def test_a003_per_probe_rows_topk_and_unchanged_without_depths(tmp_path):
    rows, ref, sd = _a3_rows(tmp_path / 'a')
    assert len(rows) == 7 and all(r['top10_identical'] and r['top50_identical'] for r in rows)
    assert all(r['first_differing_rank_0based'] is None and 7.3e-6 < r['max_rel'] < 7.5e-6 for r in rows)
    plain = F.per_probe_rows(sd, ref)                      # without rank_depths: the 002 keys only
    assert all('top50_identical' not in r and 'first_differing_rank_0based' not in r for r in plain)
    assert [{k: v for k, v in r.items() if k in plain[0]} for r in rows] == plain


def test_a003_ot_gate_passes_at_7p4e6_with_top50_identical(tmp_path):
    rows, _, _ = _a3_rows(tmp_path / 'a')
    stage, prov, census = _a3_stage(tmp_path)
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert g['pass'] and g['failures'] == []
    assert set(g['checks']) == {'max_rel_diff_lt_1e-5', 'per_row_max_rel_lt_1e-5', 'per_row_top50_identical',
                                'n_probe_rows_7', 'n_compared_cells_20440', 'provenance_exactly_7',
                                'no_FAILED_stdout_line', 'no_Traceback', 'no_numItermax_with_failed_bar',
                                'per_row_no_nan', 'per_row_n_live_eq_published', 'sentinel_pattern_identical',
                                'cli_outcome_in_amended_waiver'}
    assert g['cli_fidelity_line'] == 'FAIL' and g['cli_rc'] == 1
    assert g['cli_verdict_superseded_by_amendment_003'] is True and g['per_row_top10_identical'] is True
    assert g['amendment'] == A3_AMEND and g['amended_bar'] == {'max_rel_lt': 1e-5, 'top_k_identical': 50, 'pass': True}
    assert all('row_pickle_bytes' in r for r in g['per_row'])            # consumed by gates.py
    # the same evidence without the amendment fails exactly as fidelity_001 did (behaviour unchanged)
    o = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows)
    assert not o['pass'] and o['failures'] == ['M1.3 ot: fidelity_line_pass', 'M1.3 ot: rc_0',
                                               'M1.3 ot: max_rel_diff_lt_1e-6']
    assert 'cli_verdict_superseded_by_amendment_003' not in o and 'amendment' not in o
    # numItermax is not a STOP while the amended bar holds
    stage2, prov2, census2 = _a3_stage(tmp_path / 'n', nit=True)
    g = F.probe_gate('ot', _ot_cli(), prov2, A3_IDX, [], census2, rows, amendment=A3_AMEND)
    assert g['pass'] and g['numItermax_elements']


@pytest.mark.parametrize('cli_rel,scale,bad', [
    (1e-5, 7.4e-6, 'max_rel_diff_lt_1e-5'),                      # overall bar is strict (< 1e-5)
    (2.3e-5, 7.4e-6, 'max_rel_diff_lt_1e-5'),
    (7.4e-6, 1.2e-5, 'per_row_max_rel_lt_1e-5'),                 # per-row bar
    (None, 7.4e-6, 'max_rel_diff_lt_1e-5')])
def test_a003_ot_gate_fails_at_or_above_1e5(tmp_path, cli_rel, scale, bad):
    rows, _, _ = _a3_rows(tmp_path / 'a', scale=scale)
    stage, prov, census = _a3_stage(tmp_path, nit=True)
    if cli_rel is None:                                          # CLI JSON without max_rel_diff: not a pass
        o = _probe_obj(method='ot_cosine')
        del o['max_rel_diff']
        txt = lsf_text(cli_block(o, 'FAIL'), 'Exited with exit code 1.')
    else:
        txt = _ot_cli(max_rel=cli_rel)
    g = F.probe_gate('ot', txt, prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert not g['pass'] and f'M1.3 ot: {bad}' in g['failures']
    assert g['cli_verdict_superseded_by_amendment_003'] is False and g['amended_bar']['pass'] is False
    # rule 5: with the amended bar failed, the CLI FAIL line / rc 1 gate again; numItermax becomes a STOP
    assert {'M1.3 ot: fidelity_line_pass', 'M1.3 ot: rc_0',
            'M1.3 ot: no_numItermax_with_failed_bar'} <= set(g['failures'])


def test_a003_ot_gate_fails_when_top50_order_differs(tmp_path):
    rows, _, _ = _a3_rows(tmp_path / 'a', tie_swap_rank=20)
    assert rows[0]['top10_identical'] and not rows[0]['top50_identical']
    assert rows[0]['first_differing_rank_0based'] == 20 and rows[0]['max_rel'] < 1e-5
    stage, prov, census = _a3_stage(tmp_path)
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert not g['pass'] and 'M1.3 ot: per_row_top50_identical' in g['failures']
    assert g['checks']['max_rel_diff_lt_1e-5'] and g['checks']['per_row_max_rel_lt_1e-5']
    rows, _, _ = _a3_rows(tmp_path / 'b', tie_swap_rank=5)            # inside the top 10: top10 recorded False
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert not g['pass'] and g['per_row_top10_identical'] is False
    rows, _, _ = _a3_rows(tmp_path / 'c', tie_swap_rank=55)           # beyond depth 50: recorded, not gating
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert g['pass'] and g['per_row'][0]['first_differing_rank_0based'] == 55


def test_a003_ot_gate_per_row_bar_not_vacuous(tmp_path):
    rows, _, _ = _a3_rows(tmp_path / 'a')
    stage, prov, census = _a3_stage(tmp_path)
    for rr in ([], rows[:6]):
        g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rr, amendment=A3_AMEND)
        assert not g['pass'] and not g['checks']['per_row_max_rel_lt_1e-5']
        assert not g['checks']['per_row_top50_identical']
    no_rank = [{k: v for k, v in r.items() if not k.startswith('top')} for r in rows]
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, no_rank, amendment=A3_AMEND)
    assert not g['checks']['per_row_top50_identical']


def test_a003_bob_bar_unchanged_at_1e6(tmp_path):
    stage = str(tmp_path)
    for i in A3_IDX:
        _write_logs(stage, 'pbob', i)
    prov = os.path.join(stage, 'logs', 'similarity_virchow2_bob_probe')
    os.makedirs(prov)
    for i in A3_IDX:
        open(os.path.join(prov, f'provenance_{i}.json'), 'w').write('{}')
    rows = F.stderr_census(stage, ['pbob'])
    for rel, verdict in ((2e-6, 'PASS'), (7.4e-6, 'FAIL'), (1e-6, 'PASS')):
        g = F.probe_gate('bob', lsf_text(cli_block(_probe_obj(max_rel_diff=rel), verdict)), prov, A3_IDX, [], rows, [])
        assert not g['pass'] and 'M1.3 bob: max_rel_diff_lt_1e-6' in g['failures']
    with pytest.raises(F.FidelityError, match='ot only'):
        F.probe_gate('bob', lsf_text(cli_block(_probe_obj())), prov, A3_IDX, [], rows, [], amendment=A3_AMEND)
    assert F.main(['probe-gate', '--stage-dir', stage, '--method', 'bob', '--probes', 'x', '--element-jobname',
                   'pbob', '--combine-jobname', 'c', '--amendment-003']) == 1


def _a3_amendment(tmp_path):
    p = os.path.join(str(tmp_path), 'contract_amendment_003', 'AMENDMENT_003.md')
    os.makedirs(os.path.dirname(p))
    open(p, 'w').write('# amendment 003 (synthetic)\n')
    return p, F.sha256_file(p)


def _a3_src(tmp_path):
    src = os.path.join(str(tmp_path), 'fidelity_001')
    for rel, txt in (('m1_ot_probe.json', '{"pass": false}'), ('FIDELITY_REPORT.json', '{}'), ('FAILED', 'x'),
                     ('m1_bob_probe.json', '{"pass": true}'), ('journal.jsonl', '{}\n'),
                     ('similarity/virchow2_ot_probe/similarity_1.pkl', 'row'), ('lsf/a.1_0.out', 'out'),
                     ('jobs/m1/FAILED', 'nested name: copied'), ('queue_probe/w01/m1_ot_probe.json', 'nested')):
        os.makedirs(os.path.dirname(os.path.join(src, rel)), exist_ok=True)
        open(os.path.join(src, rel), 'w').write(txt)
    os.makedirs(os.path.join(src, 'c2_metric_suite'))          # empty directory is reproduced
    os.chmod(os.path.join(src, 'm1_bob_probe.json'), 0o500)
    return src


def _tree(d):
    out = {}
    for dp, dn, fn in os.walk(d):
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, d)] = (F.sha256_file(p), os.stat(p).st_mode)
        for n in dn:
            out[os.path.relpath(os.path.join(dp, n), d) + '/'] = None
    return out


def test_a003_init_002_lineage_and_refusal(tmp_path):
    amend, sha = _a3_amendment(tmp_path)
    src = _a3_src(tmp_path)
    before = _tree(src)
    dst = os.path.join(str(tmp_path), 'fidelity_002')
    lin = F.init_002(src, dst, amend, amendment_sha=sha)
    assert _tree(src) == before                                 # fidelity_001 untouched
    assert lin['amendment'] == {'path': amend, 'sha256': sha}
    assert sorted(e['path'] for e in lin['excluded']) == ['FAILED', 'FIDELITY_REPORT.json', 'm1_ot_probe.json']
    assert all(e['sha256'] == F.sha256_file(os.path.join(src, e['path'])) for e in lin['excluded'])
    for n in F.INIT_002_EXCLUDE:
        assert not os.path.lexists(os.path.join(dst, n))
    copied = {e['path']: e for e in lin['files']}
    assert set(copied) == {r for r in before if before[r] and r not in F.INIT_002_EXCLUDE}
    assert 'jobs/m1/FAILED' in copied and 'queue_probe/w01/m1_ot_probe.json' in copied   # only top level excluded
    for r, e in copied.items():
        assert e['sha256_source'] == e['sha256_dest'] == F.sha256_file(os.path.join(dst, r)) and e['equal']
        assert e['method'] == 'hardlink' and os.path.samefile(os.path.join(src, r), os.path.join(dst, r))
    assert lin['n_files'] == len(copied) == lin['n_hardlinked'] and lin['n_copied'] == 0
    assert os.path.isdir(os.path.join(dst, 'c2_metric_suite'))
    on_disk = json.load(open(os.path.join(dst, 'LINEAGE_002.json')))
    assert on_disk['files'] == lin['files'] and on_disk['source_dir'] == src
    with pytest.raises(F.FidelityError, match='already exists'):
        F.init_002(src, dst, amend, amendment_sha=sha)
    assert F.main(['init-002', '--source', src, '--dest', dst, '--amendment', amend]) == 1
    assert _tree(src) == before


def test_a003_init_002_copy_fallback_and_guards(tmp_path):
    amend, sha = _a3_amendment(tmp_path)
    src = _a3_src(tmp_path)
    before = _tree(src)

    def nolink(a, b):
        raise OSError(18, 'Invalid cross-device link')
    dst = os.path.join(str(tmp_path), 'f2_copy')
    lin = F.init_002(src, dst, amend, amendment_sha=sha, link=nolink)
    assert lin['n_copied'] == lin['n_files'] and lin['n_hardlinked'] == 0
    assert all(e['method'] == 'copy' and 'cross-device' in e['hardlink_error'] for e in lin['files'])
    assert not os.path.samefile(os.path.join(src, 'journal.jsonl'), os.path.join(dst, 'journal.jsonl'))
    assert _tree(src) == before
    bad = os.path.join(str(tmp_path), 'f2_badsha')
    with pytest.raises(F.FidelityError, match='M0.3'):
        F.init_002(src, bad, amend, amendment_sha='0' * 64)
    assert not os.path.lexists(bad)                            # nothing created before the amendment binding
    os.symlink(os.path.join(src, 'journal.jsonl'), os.path.join(src, 'lsf', 'link.out'))
    sym = os.path.join(str(tmp_path), 'f2_sym')
    with pytest.raises(F.FidelityError, match='symlink'):
        F.init_002(src, sym, amend, amendment_sha=sha)
    assert not os.path.lexists(sym)
    with pytest.raises(F.FidelityError, match='inside the source'):
        F.init_002(src, os.path.join(src, 'sub002'), amend, amendment_sha=sha)


def test_a003_default_stage_dir_is_fidelity_002(monkeypatch):
    assert F.FIDELITY_DIR == os.path.join(F.ROOT, 'fidelity_002')
    assert F.FIDELITY_DIR_001 == os.path.join(F.ROOT, 'fidelity_001')
    assert F.AMENDMENT_003 == os.path.join(F.ROOT, 'contract_amendment_003', 'AMENDMENT_003.md')
    seen = {}

    def cap(name):
        def f(a):
            seen[name] = a
            return 0
        return f
    for fn in ('cmd_bank_check', 'cmd_aggregate_gate', 'cmd_rebuild_gate', 'cmd_probe_gate', 'cmd_census',
               'cmd_ranks_precheck', 'cmd_ranks_gate', 'cmd_report', 'cmd_init_002'):
        monkeypatch.setattr(F, fn, cap(fn))
    for argv in (['bank-check'], ['aggregate-gate', '--method', 'sampler', '--jobname', 'j'],
                 ['rebuild-gate', '--method', 'sampler'],
                 ['probe-gate', '--method', 'ot', '--probes', 'p', '--element-jobname', 'e', '--combine-jobname', 'c'],
                 ['census', '--jobnames', 'j'], ['ranks-precheck'], ['ranks-gate', '--method', 'sampler'], ['report'],
                 ['init-002']):
        assert F.main(argv) == 0
    for k, a in seen.items():
        if k == 'cmd_init_002':
            assert a.source == F.FIDELITY_DIR_001 and a.dest == F.FIDELITY_DIR and a.amendment == F.AMENDMENT_003
        else:
            assert a.stage_dir == F.FIDELITY_DIR, k
    assert seen['cmd_probe_gate'].amendment_003 is False and seen['cmd_probe_gate'].amendment == F.AMENDMENT_003


def _a3_cmd_world(tmp_path, monkeypatch, scale=7.4e-6):
    rows, ref, sd = _a3_rows(tmp_path / 'r', scale=scale)
    pubm = os.path.join(str(tmp_path), 'pubm')
    os.makedirs(pubm)
    pu = os.path.join(pubm, 'combined_similarity_matrix_virchow2_ot_cosine.pkl')
    pickle.dump(ref, open(pu, 'wb'))
    monkeypatch.setattr(F, 'PUBM', pubm)
    monkeypatch.setitem(F.PUB_MATRIX, 'ot_cosine', (37, os.path.getsize(pu), F.sha256_file(pu)))
    amend, sha = _a3_amendment(tmp_path)
    monkeypatch.setattr(F, 'AMENDMENT_003_SHA', sha)
    stage, prov, _ = _a3_stage(tmp_path)
    os.makedirs(os.path.join(stage, 'similarity'))
    os.rename(sd, os.path.join(stage, 'similarity', 'virchow2_ot_probe'))
    open(os.path.join(stage, 'lsf', 'pcomb.950_0.out'), 'w').write(_ot_cli())
    probes = os.path.join(stage, 'm1_probes_derived.json')
    json.dump({'derived': [{'index_1based': i} for i in A3_IDX]}, open(probes, 'w'))
    argv = ['probe-gate', '--stage-dir', stage, '--method', 'ot', '--probes', probes, '--element-jobname', 'pot',
            '--combine-jobname', 'pcomb', '--amendment-003', '--amendment', amend]
    return stage, argv, amend, sha


def test_a003_cmd_probe_gate_amended_end_to_end(tmp_path, monkeypatch):
    stage, argv, amend, sha = _a3_cmd_world(tmp_path, monkeypatch)
    assert F.main(argv) == 1                                   # LINEAGE_002.json missing: refused
    assert not os.path.exists(os.path.join(stage, 'm1_ot_probe.json'))
    open(os.path.join(stage, 'LINEAGE_002.json'), 'w').write('{}\n')
    monkeypatch.setattr(F, 'FIDELITY_DIR_001', stage)          # the frozen 001 stage is refused outright
    assert F.main(argv) == 1 and not os.path.exists(os.path.join(stage, 'm1_ot_probe.json'))
    monkeypatch.setattr(F, 'FIDELITY_DIR_001', os.path.join(str(tmp_path), 'fidelity_001'))
    assert F.main(argv) == 0
    g = json.load(open(os.path.join(stage, 'm1_ot_probe.json')))
    assert g['pass'] and g['amendment'] == {'path': amend, 'sha256': sha}
    assert g['cli_verdict_superseded_by_amendment_003'] is True and g['cli_rc'] == 1
    assert g['input_bindings']['lineage_002']['sha256'] == F.sha256_file(os.path.join(stage, 'LINEAGE_002.json'))
    assert all(r['top50_identical'] and r['row_pickle_bytes'] > 0 for r in g['per_row'])
    assert F.main(argv) == 1                                   # never overwrites
    os.remove(os.path.join(stage, 'm1_ot_probe.json'))
    monkeypatch.setattr(F, 'AMENDMENT_003_SHA', '0' * 64)      # amendment bytes must match the bound sha
    assert F.main(argv) == 1 and not os.path.exists(os.path.join(stage, 'm1_ot_probe.json'))


# ----------------------------------------------------------------------------------------------
# implementation_005: impl_004 review C1/F1 (copy mode, lineage-check, report), C2/F2 (waiver), F3 (sentinel)
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name,txt_fn', [
    ('rc137_FAIL', lambda: _ot_cli(status='Exited with exit code 137.')),
    ('rc137_TERM', lambda: _ot_cli(status='TERM_MEMLIMIT: job killed after reaching LSF memory usage limit.\n'
                                          'Exited with exit code 137.')),
    ('PASS_rc1', lambda: _ot_cli(verdict='PASS')),
    ('FAIL_rc0', lambda: _ot_cli(status='Successfully completed.')),
    ('PASS_rc0', lambda: _ot_cli(verdict='PASS', status='Successfully completed.')),
    ('no_epilogue', lambda: cli_block(_probe_obj(method='ot_cosine', max_rel_diff=7.4e-6), 'FAIL') + '\n'),
    ('FAIL_rc2', lambda: _ot_cli(status='Exited with exit code 2.'))])
def test_i005_waiver_only_for_fail_rc1(tmp_path, name, txt_fn):
    rows, _, _ = _a3_rows(tmp_path / 'a')
    stage, prov, census = _a3_stage(tmp_path)
    g = F.probe_gate('ot', txt_fn(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert g['amended_bar']['pass'] is True                     # the bar holds in every case
    assert g['checks']['cli_outcome_in_amended_waiver'] is False
    assert not g['pass'] and g['cli_verdict_superseded_by_amendment_003'] is False
    assert 'M1.3 ot: cli_outcome_in_amended_waiver' in g['failures']
    assert 'fidelity_line_pass' in g['checks'] and 'rc_0' in g['checks']      # the CLI outcome gates as before
    if name in ('rc137_FAIL', 'rc137_TERM', 'FAIL_rc2', 'PASS_rc1', 'no_epilogue'):
        assert 'M1.3 ot: rc_0' in g['failures']
    if name == 'rc137_TERM':
        assert g['cli_term_reason'] == 'TERM_MEMLIMIT' and g['cli_rc'] == 137
    if name == 'no_epilogue':
        assert g['cli_epilogue'] is False and g['cli_rc'] is None


def test_i005_waiver_fail_rc1_is_the_only_superseded_outcome(tmp_path):
    rows, _, _ = _a3_rows(tmp_path / 'a')
    stage, prov, census = _a3_stage(tmp_path)
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert g['pass'] and g['checks']['cli_outcome_in_amended_waiver'] is True
    assert 'fidelity_line_pass' not in g['checks'] and 'rc_0' not in g['checks']
    assert g['cli_epilogue'] is True and g['cli_term_reason'] is None and g['cli_rc'] == 1
    assert g['cli_verdict_superseded_by_amendment_003'] is True
    no_line = lsf_text(json.dumps(_probe_obj(method='ot_cosine', max_rel_diff=7.4e-6), indent=2),
                       'Exited with exit code 1.')                  # missing FIDELITY line: STOP, nothing passes
    with pytest.raises(F.FidelityError, match='FIDELITY line'):
        F.probe_gate('ot', no_line, prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    two = _ot_cli().replace('FIDELITY: FAIL', 'FIDELITY: FAIL\nFIDELITY: FAIL')
    with pytest.raises(F.FidelityError, match='exactly one'):
        F.probe_gate('ot', two, prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    # unamended path: no waiver key, behaviour unchanged
    o = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows)
    assert 'cli_outcome_in_amended_waiver' not in o['checks'] and 'sentinel_pattern_identical' not in o['checks']


def _a3_rows_sentinel(tmp_path, how):
    """Like _a3_rows (bar holds, top-50 identical) with one sentinel/NaN-pattern defect in the first probe row."""
    rng = np.random.default_rng(11)
    n = 70
    ids = [str(1000 + i) for i in range(n)]
    M = rng.uniform(0.2, 1.0, size=(n, n))
    np.fill_diagonal(M, 1e6)
    R = np.where(M < 1e6, M * (1 + 7.4e-6), M)
    if how == 'self_live':                       # self cell recomputed as 0.0 (live) vs published sentinel
        R[0, 0] = 0.0
    elif how == 'extra_sentinel':                # recomputed sentinel where published is live (the largest cell)
        R[0, int(np.argmax(np.where(M[0] < 1e6, M[0], -1)))] = 1e6
    elif how == 'nan_both':                      # NaN in both rows at the same cell: pattern equal, no_nan fails
        M[0, 5] = np.nan
        R[0, 5] = np.nan
    elif how == 'nan_recomputed':
        R[0, 69] = np.nan
    ref = pd.DataFrame(M, index=[int(i) for i in ids], columns=[int(i) for i in ids])
    sd = os.path.join(str(tmp_path), 'sim')
    os.makedirs(sd)
    for qi in range(7):
        pickle.dump([(ids[j], float(R[qi, j])) for j in range(n)], open(os.path.join(sd, f'similarity_{ids[qi]}.pkl'),
                                                                        'wb'))
    return F.per_probe_rows(sd, ref, rank_depths=(10, 50))


@pytest.mark.parametrize('how', ['self_live', 'extra_sentinel', 'nan_recomputed'])
def test_i005_sentinel_pattern_mismatch_fails(tmp_path, how):
    rows = _a3_rows_sentinel(tmp_path / 'a', how)
    stage, prov, census = _a3_stage(tmp_path)
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert not g['pass'] and 'M1.3 ot: sentinel_pattern_identical' in g['failures']
    if how == 'self_live':
        # the F3 scenario: every other amended check passes, only the sentinel pattern catches it
        r0 = rows[0]
        assert r0['n_live_eq_published'] and r0['top50_identical'] and r0['max_rel'] < 1e-5
        assert g['failures'] == ['M1.3 ot: sentinel_pattern_identical']


def test_i005_sentinel_pattern_nan_both_and_clean(tmp_path):
    rows = _a3_rows_sentinel(tmp_path / 'a', 'nan_both')
    assert rows[0]['nan_pattern_equal'] is True and rows[0]['no_nan'] is False
    stage, prov, census = _a3_stage(tmp_path)
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert not g['pass'] and 'M1.3 ot: per_row_no_nan' in g['failures']
    rows = _a3_rows_sentinel(tmp_path / 'b', 'none')
    assert all(r['sentinel_pattern_equal'] and r['nan_pattern_equal'] for r in rows)
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, rows, amendment=A3_AMEND)
    assert g['pass'] and g['checks']['sentinel_pattern_identical'] is True
    no_key = [{k: v for k, v in r.items() if k != 'nan_pattern_equal'} for r in rows]     # fail closed
    g = F.probe_gate('ot', _ot_cli(), prov, A3_IDX, [], census, no_key, amendment=A3_AMEND)
    assert not g['checks']['sentinel_pattern_identical']


def test_i005_init_002_copy_mode_lineage(tmp_path):
    amend, sha = _a3_amendment(tmp_path)
    src = _a3_src(tmp_path)
    before = _tree(src)
    called = []

    def spy(a, b):
        called.append((a, b))
        raise AssertionError('no hardlink in copy mode')
    dst = os.path.join(str(tmp_path), 'fidelity_002')
    lin = F.init_002(src, dst, amend, amendment_sha=sha, link=spy, copy=True)
    assert not called and _tree(src) == before
    assert lin['mode'] == 'copy' and lin['n_copied'] == lin['n_files'] > 0 and lin['n_hardlinked'] == 0
    for e in lin['files']:
        assert e['mode'] == 'copy' and e['method'] == 'copy' and 'hardlink_error' not in e
        assert e['sha256_source'] == e['sha256_dest'] == F.sha256_file(os.path.join(dst, e['path']))
        assert not os.path.samefile(os.path.join(src, e['path']), os.path.join(dst, e['path']))
    assert 'copied (no hardlinks' in lin['note']
    assert json.load(open(os.path.join(dst, 'LINEAGE_002.json')))['mode'] == 'copy'
    # a write through the copy does not reach fidelity_001; lineage-check detects it
    os.chmod(os.path.join(dst, 'journal.jsonl'), 0o600)
    open(os.path.join(dst, 'journal.jsonl'), 'a').write('{"x": 1}\n')
    assert _tree(src) == before
    lc = F.check_lineage(dst)
    assert not lc['pass'] and [(m['role'], m['path']) for m in lc['mismatches']] == [('dest', 'journal.jsonl')]
    # CLI flag
    d2 = os.path.join(str(tmp_path), 'f2_cli')
    assert F.main(['init-002', '--source', src, '--dest', d2, '--amendment', amend, '--copy']) == 1   # sha pinned
    assert not os.path.lexists(d2)


def test_i005_init_002_default_mode_unchanged(tmp_path):
    amend, sha = _a3_amendment(tmp_path)
    src = _a3_src(tmp_path)
    lin = F.init_002(src, os.path.join(str(tmp_path), 'f2'), amend, amendment_sha=sha)
    assert 'mode' not in lin and all('mode' not in e and e['method'] == 'hardlink' for e in lin['files'])


def _i005_lineage_world(tmp_path, copy=True, drop_gate_files=False):
    amend, sha = _a3_amendment(tmp_path)
    src = _a3_src(tmp_path)
    if drop_gate_files:                          # so that _gates() below writes only files outside the lineage
        for g in F.GATE_ORDER:
            if os.path.exists(os.path.join(src, f'{g}.json')):
                os.remove(os.path.join(src, f'{g}.json'))
    dst = os.path.join(str(tmp_path), 'fidelity_002')
    F.init_002(src, dst, amend, amendment_sha=sha, copy=copy)
    return src, dst


def test_i005_lineage_check_pass_and_each_mismatch_role(tmp_path):
    src, dst = _i005_lineage_world(tmp_path / 'w')
    lc = F.check_lineage(dst)
    n = json.load(open(os.path.join(dst, 'LINEAGE_002.json')))['n_files']
    assert lc['pass'] and lc['mismatches'] == [] and lc['n_hashed'] == 2 * n + 3
    assert lc['lineage']['sha256'] == F.sha256_file(os.path.join(dst, 'LINEAGE_002.json'))
    assert F.main(['lineage-check', '--stage-dir', dst]) == 0
    open(os.path.join(dst, 'm1_ot_probe.json'), 'w').write('{}')            # new gate outputs are not lineage files
    assert F.check_lineage(dst)['pass']
    os.remove(os.path.join(dst, 'lsf', 'a.1_0.out'))                        # missing destination file
    os.chmod(os.path.join(src, 'FAILED'), 0o600)
    open(os.path.join(src, 'FAILED'), 'a').write('y')                       # fidelity_001 FAILED changed
    os.chmod(os.path.join(src, 'm1_bob_probe.json'), 0o700)
    open(os.path.join(src, 'm1_bob_probe.json'), 'a').write(' ')            # fidelity_001 source changed
    lc = F.check_lineage(dst)
    got = sorted((m['role'], m['path']) for m in lc['mismatches'])
    assert got == [('dest', 'lsf/a.1_0.out'), ('source', 'm1_bob_probe.json'), ('source_excluded', 'FAILED')]
    assert not lc['pass'] and F.main(['lineage-check', '--stage-dir', dst]) == 1
    with pytest.raises(F.FidelityError, match='missing'):
        F.check_lineage(os.path.join(str(tmp_path), 'w'))                    # no LINEAGE_002.json
    assert F.main(['lineage-check', '--stage-dir', src]) == 1


def test_i005_lineage_check_hardlink_write_through_detected(tmp_path):
    src, dst = _i005_lineage_world(tmp_path / 'w', copy=False)
    assert F.check_lineage(dst)['pass']
    os.chmod(os.path.join(dst, 'journal.jsonl'), 0o600)
    open(os.path.join(dst, 'journal.jsonl'), 'a').write('{"x": 1}\n')      # write-through via the shared inode
    got = sorted((m['role'], m['path']) for m in F.check_lineage(dst)['mismatches'])
    assert got == [('dest', 'journal.jsonl'), ('source', 'journal.jsonl')]


def test_i005_report_runs_lineage_check_and_refuses_on_mismatch(tmp_path):
    src, dst = _i005_lineage_world(tmp_path / 'w', drop_gate_files=True)
    _gates(dst)
    assert F.check_lineage(dst)['pass']
    os.chmod(os.path.join(dst, 'jobs', 'm1', 'FAILED'), 0o600)
    open(os.path.join(dst, 'jobs', 'm1', 'FAILED'), 'a').write('tampered')
    assert F.main(['report', '--stage-dir', dst]) == 1
    for n in ('COMPLETE', 'FAILED', 'FIDELITY_REPORT.json'):
        assert not os.path.lexists(os.path.join(dst, n))                  # refused: nothing written
    # a clean fidelity_002 gets COMPLETE with the lineage check recorded in the report
    src2, dst2 = _i005_lineage_world(tmp_path / 'v', drop_gate_files=True)
    _gates(dst2)
    assert F.main(['report', '--stage-dir', dst2]) == 0
    rep = json.load(open(os.path.join(dst2, 'FIDELITY_REPORT.json')))
    assert rep['lineage_check']['pass'] is True and rep['lineage_check']['mismatches'] == []
    assert rep['all_pass'] and os.path.exists(os.path.join(dst2, 'COMPLETE'))
    # a stage named fidelity_002 without LINEAGE_002.json is refused, not reported
    d3 = os.path.join(str(tmp_path), 'x', 'fidelity_002')
    os.makedirs(d3)
    _gates(d3)
    assert F.main(['report', '--stage-dir', d3]) == 1 and not os.path.exists(os.path.join(d3, 'COMPLETE'))
    # stages without lineage (e.g. the frozen-layout fidelity_001 tests) are unchanged: no lineage_check key
    d4 = os.path.join(str(tmp_path), 'plain')
    os.makedirs(d4)
    _gates(d4)
    assert F.main(['report', '--stage-dir', d4]) == 0
    assert 'lineage_check' not in json.load(open(os.path.join(d4, 'FIDELITY_REPORT.json')))


# ----------------------------------------------------------------------------------------------
# AMENDMENT_004: init-run-002 (run_002 from frozen FAILED run_001) and lineage-check --lineage-file
# ----------------------------------------------------------------------------------------------

A4_EXCLUDED = ['FAILED', 'host_census.tsv', 'host_sidecar/v2m_m3_gates.332494769_0.json', 'jobs/m3_gates/job.sh',
               'jobs/m3_gates/launch.sha256', 'jobs/m3_gates/readback_332494769.json', 'jobs/m3_gates/step.json',
               'lsf/v2m_m3_gates.332494769_0.err', 'lsf/v2m_m3_gates.332494769_0.out', 'm3_matrix_gates.json',
               'run_resources.json', 'stderr_census.tsv']
A4_COPIED = ['journal.jsonl', 'monitor.log']


def _a4_amendment(tmp_path):
    p = os.path.join(str(tmp_path), 'contract_amendment_004', 'AMENDMENT_004.md')
    os.makedirs(os.path.dirname(p))
    open(p, 'w').write('# amendment 004 (synthetic)\n')
    return p, F.sha256_file(p)


def _a4_src(tmp_path, failed=True):
    src = os.path.join(str(tmp_path), 'run_001')
    files = {'m3_matrix_gates.json': '{"pass": false}', 'stderr_census.tsv': 'job\n', 'host_census.tsv': 'job\n',
             'run_resources.json': '{}', 'journal.jsonl': '{"event": "submit", "step": "m3_gates"}\n',
             'monitor.log': 't\tRUN\n', 'ot_fanout_memory_decision.json': '{"fanout_mem_mb": 32768}',
             'jobs/m3_gates/job.sh': 'echo gates', 'jobs/m3_gates/step.json': '{}',
             'jobs/m3_gates/launch.sha256': 'x', 'jobs/m3_gates/readback_332494769.json': '{}',
             'jobs/m3_ot_fanout/job.sh': 'echo ot', 'jobs/m3_ot_fanout/readback_900.json': '{}',
             'jobs/m3_gates_like/FAILED': 'nested name: carried over',
             'lsf/v2m_m3_gates.332494769_0.out': 'Exited with exit code 1.', 'lsf/v2m_m3_gates.332494769_0.err':
             'STOP: M3.5.3/4 sampler: feature_dim_1280\n', 'lsf/v2m_m3_gatesx.1_0.out': 'not the gate job',
             'lsf/v2m_m3_ot_fanout.900_3.out': 'Successfully completed.', 'lsf/v2m_m3_ot_fanout.900_3.err': '',
             'host_sidecar/v2m_m3_gates.332494769_0.json': '{}', 'host_sidecar/v2m_m3_ot_fanout.900_3.json': '{}',
             'logs/aggregate/provenance_virchow2_fp32_sampler.json': '{"feature_dim": 12800}',
             'similarity/virchow2_fp32_ot/similarity_1.pkl': 'row', 'queue_probe/w01/wave.json': '{}',
             'similarity_matrix/combined_similarity_matrix_virchow2_fp32_sampler.pkl': 'matrix'}
    if failed:
        files['FAILED'] = '{"first_failing_gate": "M3.5.3/4 sampler: feature_dim_1280"}\n'
    for rel, txt in files.items():
        os.makedirs(os.path.dirname(os.path.join(src, rel)), exist_ok=True)
        open(os.path.join(src, rel), 'w').write(txt)
    os.makedirs(os.path.join(src, 'import_closure'))                  # empty directory is reproduced
    os.chmod(os.path.join(src, 'ot_fanout_memory_decision.json'), 0o444)
    return src


def _a4_world(tmp_path, **kw):
    amend, sha = _a4_amendment(tmp_path)
    src = _a4_src(tmp_path)
    dst = os.path.join(str(tmp_path), 'run_002')
    lin = F.init_run_002(src, dst, amend, amendment_sha=sha, **kw)
    return src, dst, amend, sha, lin


def test_a004_constants():
    assert F.RUN_DIR_001 == os.path.join(F.ROOT, 'run_001') and F.RUN_DIR_002 == os.path.join(F.ROOT, 'run_002')
    assert F.AMENDMENT_004 == os.path.join(F.ROOT, 'contract_amendment_004', 'AMENDMENT_004.md')
    assert F.AMENDMENT_004_SHA == 'c8bcc7b36da4dd0b00ec80e4e2f34f4ac5419b1b284e223cd3e95c89f00fc2ff'
    assert F.LINEAGE_RUN_002 == 'LINEAGE_RUN_002.json' and F.LINEAGE_002 == 'LINEAGE_002.json'
    assert 'journal.jsonl' in F.INIT_RUN_002_COPY and 'journal.jsonl' not in F.INIT_RUN_002_EXCLUDE
    assert not any('332494769' in p for p in F.INIT_RUN_002_EXCLUDE)    # job id matched by pattern only


def test_a004_init_run_002_exclusion_inodes_and_lineage(tmp_path):
    amend, sha = _a4_amendment(tmp_path)
    src = _a4_src(tmp_path)
    before = _tree(src)
    dst = os.path.join(str(tmp_path), 'run_002')
    lin = F.init_run_002(src, dst, amend, amendment_sha=sha)
    assert _tree(src) == before                                     # run_001 untouched
    assert sorted(e['path'] for e in lin['excluded']) == A4_EXCLUDED
    assert all(e['sha256'] == F.sha256_file(os.path.join(src, e['path'])) for e in lin['excluded'])
    for r in A4_EXCLUDED:
        assert not os.path.lexists(os.path.join(dst, r))
    assert not os.path.lexists(os.path.join(dst, 'jobs', 'm3_gates')) and lin['excluded_dirs'] == ['jobs/m3_gates']
    assert os.path.isdir(os.path.join(dst, 'import_closure'))
    files = {e['path']: e for e in lin['files']}
    assert set(files) == {r for r in before if before[r] and r not in A4_EXCLUDED}
    assert 'jobs/m3_gates_like/FAILED' in files and 'lsf/v2m_m3_gatesx.1_0.out' in files
    for r, e in files.items():
        assert e['sha256_source'] == e['sha256_dest'] == F.sha256_file(os.path.join(dst, r)) and e['equal']
        if r in A4_COPIED:                                          # append-only state: content copy, new inode
            assert e['mode'] == 'copy' and e['copy_reason'] == 'append_only_state' and e['append_only']
            assert os.stat(os.path.join(src, r)).st_ino != os.stat(os.path.join(dst, r)).st_ino
            assert not os.path.samefile(os.path.join(src, r), os.path.join(dst, r))
        else:
            assert e['mode'] == 'hardlink' and os.path.samefile(os.path.join(src, r), os.path.join(dst, r))
    assert lin['n_copied'] == 2 and lin['n_hardlinked'] == len(files) - 2
    assert lin['amendment'] == {'path': amend, 'sha256': sha}
    assert lin['run_001_failed'] == {'path': os.path.join(src, 'FAILED'),
                                     'sha256': F.sha256_file(os.path.join(src, 'FAILED'))}
    assert lin['schema'] == F.LINEAGE_RUN_002_SCHEMA and lin['excluded_patterns'] == list(F.INIT_RUN_002_EXCLUDE)
    on_disk = json.load(open(os.path.join(dst, F.LINEAGE_RUN_002)))
    assert on_disk['files'] == lin['files'] and on_disk['source_dir'] == src and on_disk['dest_dir'] == dst
    # the copied journal is independent: an append in run_002 leaves run_001 unchanged
    open(os.path.join(dst, 'journal.jsonl'), 'a').write('{"event": "submit", "step": "m3_gates", "run": 2}\n')
    assert _tree(src) == before


def test_a004_init_run_002_refusals(tmp_path):
    amend, sha = _a4_amendment(tmp_path)
    src = _a4_src(tmp_path)
    before = _tree(src)
    dst = os.path.join(str(tmp_path), 'run_002')
    os.makedirs(dst)                                                # destination exists
    with pytest.raises(F.FidelityError, match='already exists'):
        F.init_run_002(src, dst, amend, amendment_sha=sha)
    assert os.listdir(dst) == []
    assert F.main(['init-run-002', '--source', src, '--dest', dst, '--amendment', amend]) == 1
    nofail = _a4_src(tmp_path / 'nf', failed=False)                 # no FAILED: nothing to retry
    d2 = os.path.join(str(tmp_path), 'nf_run_002')
    with pytest.raises(F.FidelityError, match='FAILED'):
        F.init_run_002(nofail, d2, amend, amendment_sha=sha)
    assert not os.path.lexists(d2)
    d3 = os.path.join(str(tmp_path), 'bad_sha_run_002')             # amendment binding checked before any write
    with pytest.raises(F.FidelityError, match='M0.3'):
        F.init_run_002(src, d3, amend, amendment_sha='0' * 64)
    assert not os.path.lexists(d3)
    with pytest.raises(F.FidelityError, match='inside the source'):
        F.init_run_002(src, os.path.join(src, 'run_002'), amend, amendment_sha=sha)
    os.symlink(os.path.join(src, 'monitor.log'), os.path.join(src, 'lsf', 'link.out'))
    d4 = os.path.join(str(tmp_path), 'sym_run_002')
    with pytest.raises(F.FidelityError, match='symlink'):
        F.init_run_002(src, d4, amend, amendment_sha=sha)
    assert not os.path.lexists(d4)
    os.remove(os.path.join(src, 'lsf', 'link.out'))
    assert _tree(src) == before


def test_a004_init_run_002_link_failure_and_copy_mode(tmp_path):
    amend, sha = _a4_amendment(tmp_path)
    src = _a4_src(tmp_path)

    def nolink(a, b):
        raise OSError(18, 'Invalid cross-device link')
    lin = F.init_run_002(src, os.path.join(str(tmp_path), 'r2_nolink'), amend, amendment_sha=sha, link=nolink)
    assert lin['n_hardlinked'] == 0 and lin['n_copied'] == lin['n_files']
    for e in lin['files']:
        assert e['mode'] == 'copy'
        if e['path'] in A4_COPIED:
            assert e['copy_reason'] == 'append_only_state' and 'hardlink_error' not in e
        else:
            assert e['copy_reason'] == 'hardlink_failed' and 'cross-device' in e['hardlink_error']
    calls = []

    def spy(a, b):
        calls.append(a)
        os.link(a, b)
    lin = F.init_run_002(src, os.path.join(str(tmp_path), 'r2_copy'), amend, amendment_sha=sha, link=spy, copy=True)
    assert calls == [] and lin['mode'] == 'copy' and all(e['mode'] == 'copy' for e in lin['files'])
    lin = F.init_run_002(src, os.path.join(str(tmp_path), 'r2_spy'), amend, amendment_sha=sha, link=spy)
    assert 'journal.jsonl' not in [os.path.relpath(c, src) for c in calls]      # journal never hardlinked
    assert len(calls) == lin['n_hardlinked'] == lin['n_files'] - 2


def test_a004_lineage_check_lineage_file(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    argv = ['lineage-check', '--stage-dir', dst, '--lineage-file', F.LINEAGE_RUN_002]
    assert F.main(argv) == 0
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002)
    assert lc['pass'] and lc['n_hashed'] == 2 * lin['n_files'] + len(A4_EXCLUDED)
    assert lc['n_excluded_dest_checked'] == len(A4_EXCLUDED) + 1 and lc['append_only_appended'] == []
    assert lc['lineage']['sha256'] == F.sha256_file(os.path.join(dst, F.LINEAGE_RUN_002))
    assert F.main(['lineage-check', '--stage-dir', dst]) == 1        # default LINEAGE_002.json: absent in run_002
    # run_002 appends to its own journal / monitor.log: accepted, reported
    open(os.path.join(dst, 'journal.jsonl'), 'a').write('{"event": "submit", "step": "m3_gates"}\n')
    open(os.path.join(dst, 'monitor.log'), 'a').write('t2\tDONE\n')
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002)
    assert lc['pass'] and sorted(a['path'] for a in lc['append_only_appended']) == A4_COPIED
    # strict mode: any excluded path present in run_002 fails; regenerated outputs pass the pre-package mode
    os.makedirs(os.path.join(dst, 'jobs', 'm3_gates'))
    open(os.path.join(dst, 'jobs', 'm3_gates', 'job.sh'), 'w').write('echo gates run_002')
    open(os.path.join(dst, 'm3_matrix_gates.json'), 'w').write('{"pass": true}')
    got = sorted((m['role'], m['path']) for m in F.check_lineage(dst, F.LINEAGE_RUN_002)['mismatches'])
    assert got == [('dest_excluded_present', 'jobs/m3_gates'), ('dest_excluded_present', 'jobs/m3_gates/job.sh'),
                   ('dest_excluded_present', 'm3_matrix_gates.json')]
    assert F.main(argv) == 1
    assert F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)['pass']
    assert F.main(argv + ['--excluded-may-be-regenerated']) == 0
    # ... but never a carried-over run_001 file (same inode or same bytes)
    os.link(os.path.join(src, 'stderr_census.tsv'), os.path.join(dst, 'stderr_census.tsv'))
    open(os.path.join(dst, 'host_census.tsv'), 'w').write(open(os.path.join(src, 'host_census.tsv')).read())
    got = sorted((m['role'], m['path']) for m in
                 F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)['mismatches'])
    assert got == [('dest_excluded_carried_over', 'host_census.tsv'),
                   ('dest_excluded_carried_over', 'stderr_census.tsv')]


def test_a004_lineage_check_mismatch_roles(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    with open(os.path.join(dst, 'journal.jsonl'), 'r+') as fh:      # the recorded journal prefix rewritten
        fh.write('X')
    os.chmod(os.path.join(src, 'ot_fanout_memory_decision.json'), 0o644)
    open(os.path.join(src, 'ot_fanout_memory_decision.json'), 'a').write(' ')   # write-through via the hardlink
    open(os.path.join(src, 'FAILED'), 'a').write(' ')                # run_001 FAILED changed
    open(os.path.join(src, 'monitor.log'), 'a').write('x')          # run_001 append-only source changed
    os.remove(os.path.join(dst, 'lsf', 'v2m_m3_ot_fanout.900_3.out'))
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002)
    got = sorted((m['role'], m['path']) for m in lc['mismatches'])
    assert got == [('dest', 'lsf/v2m_m3_ot_fanout.900_3.out'), ('dest', 'ot_fanout_memory_decision.json'),
                   ('dest_append_only', 'journal.jsonl'), ('source', 'monitor.log'),
                   ('source', 'ot_fanout_memory_decision.json'), ('source_excluded', 'FAILED')]
    assert not lc['pass']
    src2, dst2, _, _, _ = _a4_world(tmp_path / 'v')
    with open(os.path.join(dst2, 'journal.jsonl'), 'w') as fh:     # truncated
        fh.write('{')
    got = [(m['role'], m['path']) for m in F.check_lineage(dst2, F.LINEAGE_RUN_002)['mismatches']]
    assert got == [('dest_append_only', 'journal.jsonl')]


def test_a004_cli_defaults(monkeypatch):
    seen = {}
    for fn in ('cmd_init_run_002', 'cmd_lineage_check'):
        monkeypatch.setattr(F, fn, (lambda n: lambda a: seen.__setitem__(n, a) or 0)(fn))
    assert F.main(['init-run-002']) == 0 and F.main(['lineage-check']) == 0
    a = seen['cmd_init_run_002']
    assert (a.source, a.dest, a.amendment, a.copy) == (F.RUN_DIR_001, F.RUN_DIR_002, F.AMENDMENT_004, False)
    a = seen['cmd_lineage_check']
    assert (a.stage_dir, a.lineage_file, a.excluded_may_be_regenerated) == (F.FIDELITY_DIR, F.LINEAGE_002, False)


# ----------------------------------------------------------------------------------------------
# ADDENDUM_002 F1/F1b (unlisted-file walk + expected-new allowlist) and F2 (regenerated excluded names)
# ----------------------------------------------------------------------------------------------

def _put(root, rel, txt='new'):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w').write(txt)
    return p


def _allow(tmp_path, globs, name='expected_new.json', **extra):
    p = os.path.join(str(tmp_path), name)
    json.dump({'schema': 'run_002_expected_new/1', 'globs': globs, **extra}, open(p, 'w'))
    return p


def _roles(lc):
    return sorted((m['role'], m['path']) for m in lc['mismatches'])


def _regenerate_m3(dst):
    """What the run_002 m3_gates job legitimately writes: fresh files at the excluded names."""
    for rel in ('m3_matrix_gates.json', 'stderr_census.tsv', 'host_census.tsv', 'run_resources.json',
                'jobs/m3_gates/job.sh', 'jobs/m3_gates/step.json', 'jobs/m3_gates/launch.sha256',
                'jobs/m3_gates/readback_777.json', 'lsf/v2m_m3_gates.777_0.out', 'lsf/v2m_m3_gates.777_0.err',
                'host_sidecar/v2m_m3_gates.777_0.json'):
        _put(dst, rel, f'run_002 {rel}')


@pytest.mark.parametrize('pat,rel,want', [
    ('FAILED', 'FAILED', True), ('FAILED', 'x/FAILED', False), ('FAILED', 'FAILEDx', False),
    ('jobs/m3_gates/**', 'jobs/m3_gates/job.sh', True), ('jobs/m3_gates/**', 'jobs/m3_gates/a/b/c', True),
    ('jobs/m3_gates/**', 'jobs/m3_gates', False), ('jobs/m3_gates/**', 'jobs/m3_gates_like/FAILED', False),
    ('lsf/v2m_m3_gates.*', 'lsf/v2m_m3_gates.1_0.out', True), ('lsf/v2m_m3_gates.*', 'lsf/v2m_m3_gatesx.1_0.out',
                                                               False),
    ('lsf/v2m_m3_gates.*', 'lsf/sub/v2m_m3_gates.1', False),       # '*' never crosses '/'
    ('queue_probe/*_r002/**', 'queue_probe/w16384_60_r002/wave.json', True),
    ('queue_probe/*_r002/**', 'queue_probe/w16384_60/wave.json', False),
    ('queue_probe/*_r002/**', 'queue_probe/w1_r002', False),
    ('c2_metric_suite/**', 'c2_metric_suite/.hidden', True),       # fnmatch: '*' matches a leading dot
    ('a/**/b.json', 'a/b.json', True), ('a/**/b.json', 'a/x/y/b.json', True), ('a/**/b.json', 'a/x/c.json', False),
    ('**/x.json', 'x.json', True), ('**/x.json', 'p/q/x.json', True),
    ('eval_job_*.out', 'eval_job_332.out', True), ('eval_job_*.out', 'eval_job_332.err', False),
])
def test_f1b_relpath_glob_match(pat, rel, want):
    assert F.relpath_glob_match(pat, rel) is want


def test_f1b_glob_match_agrees_with_run_002_excluded():
    rels = ['FAILED', 'm3_matrix_gates.json', 'jobs/m3_gates/job.sh', 'jobs/m3_gates/x/y', 'jobs/m3_gates',
            'jobs/m3_gates_like/FAILED', 'lsf/v2m_m3_gates.1_0.out', 'lsf/v2m_m3_gatesx.1_0.out', 'lsf/FAILED',
            'host_sidecar/v2m_m3_gates.2_3.json', 'host_sidecar/v2m_m3_ot_fanout.1_1.json', 'run_resources.json/x']
    for r in rels:
        assert any(F.relpath_glob_match(p, r) for p in F.INIT_RUN_002_EXCLUDE) is F.run_002_excluded(r), r


def test_f1_unlisted_file_fails_both_modes(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002)
    assert lc['pass'] and lc['n_unlisted'] == 0 and lc['n_allowed_new'] == 0 and lc['expected_new'] is None
    _put(dst, 'stray.txt')                                         # top level
    _put(dst, 'lsf/v2m_m4_ranks_sampler.5_0.out')                 # inside an inherited directory
    _put(dst, 'jobs/m3_ot_fanout/readback_901.json')              # next to listed files
    _put(dst, 'sub/' + F.LINEAGE_RUN_002)                         # only the top-level lineage file is exempt
    want = [('unlisted_file', 'jobs/m3_ot_fanout/readback_901.json'),
            ('unlisted_file', 'lsf/v2m_m4_ranks_sampler.5_0.out'), ('unlisted_file', 'stray.txt'),
            ('unlisted_file', 'sub/' + F.LINEAGE_RUN_002)]
    for relaxed in (False, True):
        lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=relaxed)
        assert not lc['pass'] and _roles(lc) == want
        assert lc['n_unlisted'] == 4 and lc['n_allowed_new'] == 0 and lc['expected_new'] is None
        assert all(m['kind'] == 'regular' for m in lc['mismatches'])
    argv = ['lineage-check', '--stage-dir', dst, '--lineage-file', F.LINEAGE_RUN_002]
    assert F.main(argv) == 1 and F.main(argv + ['--excluded-may-be-regenerated']) == 1


def test_f1_unlisted_symlink_and_fifo(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    os.symlink(os.path.join(dst, 'monitor.log'), os.path.join(dst, 'lsf', 'alias.out'))
    os.mkfifo(os.path.join(dst, 'pipe'))
    os.makedirs(os.path.join(dst, 'empty_new_dir'))               # a bare directory is not a file
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)
    assert _roles(lc) == [('unlisted_file', 'lsf/alias.out'), ('unlisted_file', 'pipe')]
    assert sorted(m['kind'] for m in lc['mismatches']) == ['other', 'symlink'] and lc['n_unlisted'] == 2


def test_f1b_allowlist_accept_reject_and_record(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    al = _allow(tmp_path, ['m4_ranks_resolution.json', 'lsf/v2m_m4_ranks_*', 'c2_metric_suite/**',
                           'queue_probe/*_r002/**'])
    for rel in ('m4_ranks_resolution.json', 'lsf/v2m_m4_ranks_sampler.5_0.out', 'c2_metric_suite/a/b.json',
                'queue_probe/w4096_30_r002/wave.json'):
        _put(dst, rel)
    _put(dst, 'lsf/v2m_m5_compare.6_0.out')                       # not allowlisted
    _put(dst, 'queue_probe/w4096_30/wave.json')                   # wrong wave name
    for relaxed in (False, True):
        lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=relaxed, expected_new=al)
        assert _roles(lc) == [('unlisted_file', 'lsf/v2m_m5_compare.6_0.out'),
                              ('unlisted_file', 'queue_probe/w4096_30/wave.json')]
        assert lc['n_allowed_new'] == 4 and lc['n_unlisted'] == 2 and not lc['pass']
        assert lc['expected_new'] == {'path': al, 'sha256': F.sha256_file(al), 'n_globs': 4}
    os.remove(os.path.join(dst, 'lsf', 'v2m_m5_compare.6_0.out'))
    os.remove(os.path.join(dst, 'queue_probe', 'w4096_30', 'wave.json'))
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, expected_new=al)
    assert lc['pass'] and lc['n_allowed_new'] == 4
    argv = ['lineage-check', '--stage-dir', dst, '--lineage-file', F.LINEAGE_RUN_002]
    assert F.main(argv + ['--expected-new', al]) == 0
    assert F.main(argv) == 1                                     # without the allowlist nothing new is admitted


def test_f1b_allowlist_does_not_admit_excluded_names_in_strict_mode(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    _regenerate_m3(dst)
    al = _allow(tmp_path, ['m3_matrix_gates.json', 'stderr_census.tsv', 'host_census.tsv', 'run_resources.json',
                           'jobs/m3_gates/**', 'lsf/v2m_m3_gates.*', 'host_sidecar/v2m_m3_gates.*'])
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, expected_new=al)
    assert not lc['pass'] and {m['role'] for m in lc['mismatches']} == {'dest_excluded_present'}
    assert ('dest_excluded_present', 'lsf/v2m_m3_gates.777_0.out') in _roles(lc)    # new job id: pattern match
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True, expected_new=al)
    assert lc['pass'], lc['mismatches']
    assert lc['n_regenerated_new'] == 4 and lc['n_allowed_new'] == 0 and lc['n_unlisted'] == 0
    assert F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)['pass']


@pytest.mark.parametrize('doc', [
    {'schema': 'run_002_expected_new/2', 'globs': ['a']}, {'globs': ['a']}, {'schema': 'run_002_expected_new/1'},
    {'schema': 'run_002_expected_new/1', 'globs': []}, {'schema': 'run_002_expected_new/1', 'globs': 'a'},
    {'schema': 'run_002_expected_new/1', 'globs': ['a'], 'extra': 1}, ['a'],
    {'schema': 'run_002_expected_new/1', 'globs': ['']}, {'schema': 'run_002_expected_new/1', 'globs': [1]},
    {'schema': 'run_002_expected_new/1', 'globs': ['/abs']}, {'schema': 'run_002_expected_new/1', 'globs': ['a/../b']},
    {'schema': 'run_002_expected_new/1', 'globs': ['a//b']}, {'schema': 'run_002_expected_new/1', 'globs': ['./a']},
    {'schema': 'run_002_expected_new/1', 'globs': ['jobs/{a,b}/**']},
])
def test_f1b_allowlist_schema_errors(tmp_path, doc):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    al = os.path.join(str(tmp_path), 'bad.json')
    json.dump(doc, open(al, 'w'))
    with pytest.raises(F.FidelityError, match='expected-new'):
        F.check_lineage(dst, F.LINEAGE_RUN_002, expected_new=al)
    assert F.main(['lineage-check', '--stage-dir', dst, '--lineage-file', F.LINEAGE_RUN_002,
                   '--expected-new', al]) == 1


def test_f1b_allowlist_unreadable_and_non_run002_lineage(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    al = os.path.join(str(tmp_path), 'bad.json')
    open(al, 'w').write('{not json')
    with pytest.raises(F.FidelityError, match='unreadable'):
        F.check_lineage(dst, F.LINEAGE_RUN_002, expected_new=al)
    with pytest.raises(F.FidelityError, match='unreadable'):
        F.check_lineage(dst, F.LINEAGE_RUN_002, expected_new=al + '.absent')
    fd = os.path.join(str(tmp_path), 'fid_002')                   # LINEAGE_002-style record: no walk, no allowlist
    os.makedirs(fd)
    json.dump({'files': [], 'source_dir': fd, 'excluded': []}, open(os.path.join(fd, F.LINEAGE_002), 'w'))
    _put(fd, 'stray.txt')
    lc = F.check_lineage(fd)
    assert lc['pass'] and 'n_unlisted' not in lc                  # fidelity_002 behaviour unchanged
    with pytest.raises(F.FidelityError, match='applies only'):
        F.check_lineage(fd, expected_new=_allow(tmp_path, ['stray.txt']))


def test_f2_directory_or_symlink_at_excluded_or_allowlisted_name(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    other = _put(str(tmp_path), 'elsewhere.json', 'fresh')
    os.makedirs(os.path.join(dst, 'm3_matrix_gates.json'))                    # recorded excluded file -> directory
    os.symlink(other, os.path.join(dst, 'stderr_census.tsv'))                 # recorded excluded file -> symlink
    os.makedirs(os.path.join(dst, 'lsf'), exist_ok=True)
    os.symlink(other, os.path.join(dst, 'lsf', 'v2m_m3_gates.777_0.out'))     # new name under an excluded pattern
    os.makedirs(os.path.join(dst, 'lsf', 'v2m_m3_gates.777_0.err'))           # directory under an excluded pattern
    os.symlink(other, os.path.join(dst, 'jobs', 'm3_gates'))                  # excluded subtree root as symlink
    os.symlink(other, os.path.join(dst, 'm4_ranks_resolution.json'))          # allowlisted name -> symlink
    os.makedirs(os.path.join(dst, 'tie_census_host_diversity.json'))          # allowlisted name -> directory
    al = _allow(tmp_path, ['m4_ranks_resolution.json', 'tie_census_host_diversity.json'])
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True, expected_new=al)
    assert _roles(lc) == [('dest_excluded_not_regular', p) for p in (
        'jobs/m3_gates', 'lsf/v2m_m3_gates.777_0.err', 'lsf/v2m_m3_gates.777_0.out', 'm3_matrix_gates.json',
        'm4_ranks_resolution.json', 'stderr_census.tsv', 'tie_census_host_diversity.json')]
    kinds = {m['path']: m['kind'] for m in lc['mismatches']}
    assert kinds['m3_matrix_gates.json'] == 'directory' and kinds['stderr_census.tsv'] == 'symlink'


def test_f2_excluded_directory_subtree_files_checked_individually(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    _regenerate_m3(dst)
    os.makedirs(os.path.join(dst, 'jobs', 'm3_gates', 'nested'))
    os.symlink(os.path.join(dst, 'monitor.log'), os.path.join(dst, 'jobs', 'm3_gates', 'nested', 'l.txt'))
    os.remove(os.path.join(dst, 'jobs', 'm3_gates', 'step.json'))
    os.link(os.path.join(src, 'jobs', 'm3_ot_fanout', 'job.sh'), os.path.join(dst, 'jobs', 'm3_gates', 'step.json'))
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)
    assert _roles(lc) == [('dest_excluded_carried_over', 'jobs/m3_gates/step.json'),
                          ('dest_excluded_not_regular', 'jobs/m3_gates/nested/l.txt')]


def test_f2_second_hardlink_to_other_run_001_file_rejected(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    _regenerate_m3(dst)
    assert F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)['pass']
    # recorded excluded name hardlinked to a DIFFERENT run_001 file (not its counterpart, different bytes)
    os.remove(os.path.join(dst, 'm3_matrix_gates.json'))
    os.link(os.path.join(src, 'logs', 'aggregate', 'provenance_virchow2_fp32_sampler.json'),
            os.path.join(dst, 'm3_matrix_gates.json'))
    # new name under an excluded pattern hardlinked to a run_001 fan-out log
    os.remove(os.path.join(dst, 'lsf', 'v2m_m3_gates.777_0.out'))
    os.link(os.path.join(src, 'lsf', 'v2m_m3_ot_fanout.900_3.out'), os.path.join(dst, 'lsf', 'v2m_m3_gates.777_0.out'))
    # new excluded-pattern name sharing an inode with a file outside run_001 (any second link is refused)
    os.remove(os.path.join(dst, 'host_sidecar', 'v2m_m3_gates.777_0.json'))
    twin = _put(str(tmp_path), 'twin.json', 'fresh sidecar')
    os.link(twin, os.path.join(dst, 'host_sidecar', 'v2m_m3_gates.777_0.json'))
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)
    assert _roles(lc) == [('dest_excluded_carried_over', 'host_sidecar/v2m_m3_gates.777_0.json'),
                          ('dest_excluded_carried_over', 'lsf/v2m_m3_gates.777_0.out'),
                          ('dest_excluded_carried_over', 'm3_matrix_gates.json')]
    acts = {m['path']: m['actual'] for m in lc['mismatches']}
    assert acts['host_sidecar/v2m_m3_gates.777_0.json'].startswith('st_nlink=2 ')
    assert acts['m3_matrix_gates.json'].startswith('st_nlink=3 ')  # run_001 file is also hardlinked in run_002
    assert acts['lsf/v2m_m3_gates.777_0.out'].startswith('st_nlink=3 ')
    # the run_001 counterpart inode itself (new job-id name absent in run_001, recorded name present)
    os.remove(os.path.join(dst, 'm3_matrix_gates.json'))
    os.link(os.path.join(src, 'm3_matrix_gates.json'), os.path.join(dst, 'm3_matrix_gates.json'))
    m = [x for x in F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)['mismatches']
         if x['path'] == 'm3_matrix_gates.json'][0]
    assert m['role'] == 'dest_excluded_carried_over' and m['actual'] == 'run_001 inode'


def test_f1_f2_full_run_002_sequence_with_allowlist(tmp_path):
    src, dst, amend, sha, lin = _a4_world(tmp_path / 'w')
    _regenerate_m3(dst)
    open(os.path.join(dst, 'journal.jsonl'), 'a').write('{"event": "submit", "step": "m4_ranks_sampler"}\n')
    for rel in ('queue_probe/w4096_30_r002/wave.json', 'c2_metric_suite/metrics_virchow2_fp32_sampler.json',
                'm4_ranks_resolution.json', 'jobs/m4_ranks_sampler/job.sh', 'lsf/v2m_m4_ranks_sampler.8_0.out',
                'host_sidecar/v2m_m4_ranks_sampler.8_0.json', 'eval_attempt_001/x/y.csv', 'COMPLETE.json',
                'SHA256SUMS', 'RUN_002_EXPECTED_NEW.json'):
        _put(dst, rel)
    al = _allow(tmp_path, ['m3_matrix_gates.json', 'stderr_census.tsv', 'host_census.tsv', 'run_resources.json',
                           'jobs/m3_gates/**', 'lsf/v2m_m3_gates.*', 'host_sidecar/v2m_m3_gates.*',
                           'queue_probe/*_r002/**', 'c2_metric_suite/**', 'm4_ranks_resolution.json',
                           'jobs/m4_ranks_sampler/**', 'lsf/v2m_m4_ranks_sampler.*',
                           'host_sidecar/v2m_m4_ranks_sampler.*', 'eval_attempt_001/**', 'COMPLETE.json',
                           'SHA256SUMS', 'RUN_002_EXPECTED_NEW.json'])
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True, expected_new=al)
    assert lc['pass'], lc['mismatches']
    assert (lc['n_allowed_new'], lc['n_regenerated_new'], lc['n_unlisted']) == (10, 4, 0)
    assert lc['append_only_appended'][0]['path'] == 'journal.jsonl'
    _put(dst, 'eval_attempt_002/z.csv')                           # an unplanned second attempt is caught
    lc = F.check_lineage(dst, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True, expected_new=al)
    assert _roles(lc) == [('unlisted_file', 'eval_attempt_002/z.csv')]


def test_f1b_cli_expected_new_default_none(monkeypatch):
    seen = {}
    monkeypatch.setattr(F, 'cmd_lineage_check', lambda a: seen.__setitem__('a', a) or 0)
    assert F.main(['lineage-check']) == 0 and seen['a'].expected_new is None
    assert F.main(['lineage-check', '--expected-new', 'x.json']) == 0 and seen['a'].expected_new == 'x.json'


# ----------------------------------------------------------------------------------------------
# ADDENDUM_003 (implementation_008): init-run (run_003 from frozen FAILED run_002) and run-lineage checks
# ----------------------------------------------------------------------------------------------

A8_EXCLUDE = ('FAILED', 'm4_ranks_resolution.json', 'stderr_census.tsv', 'host_census.tsv', 'run_resources.json',
              'jobs/m4_ranks_gate/**', 'lsf/v2m_m4_ranks_gate.*', 'host_sidecar/v2m_m4_ranks_gate.*',
              'import_closure/v2m_m4_ranks_gate.*')
A8_EXCLUDED = ['FAILED', 'host_census.tsv', 'host_sidecar/v2m_m4_ranks_gate.333_0.json',
               'import_closure/v2m_m4_ranks_gate.333_0.json', 'jobs/m4_ranks_gate/job.sh',
               'jobs/m4_ranks_gate/launch.sha256', 'jobs/m4_ranks_gate/step.json', 'lsf/v2m_m4_ranks_gate.333_0.err',
               'lsf/v2m_m4_ranks_gate.333_0.out', 'm4_ranks_resolution.json', 'run_resources.json',
               'stderr_census.tsv']


def _a8_addendum(tmp_path):
    p = os.path.join(str(tmp_path), 'contract_amendment_004', 'ADDENDUM_003.md')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w').write('# addendum 003 (synthetic)\n')
    return p, F.sha256_file(p)


def _a8_run002(tmp_path, failed=True):
    """A FAILED run_002 (made by the real init_run_002 from a synthetic FAILED run_001) holding the run_002 outputs
    of the real sequence: m3 regenerated, m4_ranks_* outputs, an m4_ranks_gate attempt, probe waves, appends."""
    src, rd2, amend, sha, lin = _a4_world(tmp_path)
    _regenerate_m3(rd2)
    for rel, txt in {'queue_probe/w4096_30_r002/wave.json': '{"stage": "run_002"}',
                     'c2_metric_suite/metrics_virchow2_fp32_sampler.json': '{"config": {}}',
                     'c2_metric_suite/ranked_lists_virchow2_fp32_sampler.csv.gz': 'ranks',
                     'jobs/m4_ranks_sampler/job.sh': 'echo ranks', 'lsf/v2m_m4_ranks_sampler.8_0.out': 'ok',
                     'host_sidecar/v2m_m4_ranks_sampler.8_0.json': '{}',
                     'jobs/m4_ranks_gate/job.sh': 'echo gate', 'jobs/m4_ranks_gate/step.json': '{}',
                     'jobs/m4_ranks_gate/launch.sha256': 'x', 'lsf/v2m_m4_ranks_gate.333_0.out': 'Exited 1',
                     'lsf/v2m_m4_ranks_gate.333_0.err': 'STOP: M4.1 median_pooling: x',
                     'host_sidecar/v2m_m4_ranks_gate.333_0.json': '{}',
                     'import_closure/v2m_m4_ranks_gate.333_0.json': '{}',
                     'import_closure/v2m_m4_ranks_sampler.8_0.json': '{}',
                     'jobs/m5_tie_hosts/job.sh': 'echo tie (prepared, never submitted)',
                     'm4_ranks_resolution.json': '{"pass": false}', 'RUN_002_EXPECTED_NEW.json': '{}'}.items():
        _put(rd2, rel, txt)
    open(os.path.join(rd2, 'journal.jsonl'), 'a').write('{"event": "submit", "step": "m4_ranks_gate"}\n')
    if failed:
        _put(rd2, 'FAILED', '{"first_failing_gate": "M4.1 median_pooling: resolved_is_fp32_run_matrix"}\n')
    return src, rd2


def _a8_world(tmp_path, **kw):
    src, rd2 = _a8_run002(tmp_path)
    add, sha = _a8_addendum(tmp_path)
    rd3 = os.path.join(str(tmp_path), 'run_003')
    lin = F.init_run_003(rd2, rd3, add, addendum_sha=sha, **kw)
    return src, rd2, rd3, add, sha, lin


def test_a8_constants():
    assert F.RUN_DIR_003 == os.path.join(F.ROOT, 'run_003')
    assert F.ADDENDUM_003 == os.path.join(F.ROOT, 'contract_amendment_004', 'ADDENDUM_003.md')
    assert F.ADDENDUM_003_SHA == '553c189d2766772eecf71fa8e60b4dde35388118d87fe8e0ef696c84a110b1e3'
    assert F.LINEAGE_RUN_003 == 'LINEAGE_RUN_003.json'
    assert F.LINEAGE_RUN_003_SCHEMA == 'r2_1_virchow2_fp32_main.lineage_run_003.v1'
    assert F.INIT_RUN_003_EXCLUDE == A8_EXCLUDE                    # the ADDENDUM_003 §2.2 list, literally
    assert F.INIT_RUN_003_COPY == F.INIT_RUN_002_COPY
    assert F.RUN_LINEAGES == {'run_002': (F.LINEAGE_RUN_002, F.LINEAGE_RUN_002_SCHEMA),
                              'run_003': (F.LINEAGE_RUN_003, F.LINEAGE_RUN_003_SCHEMA)}
    if os.path.exists(F.ADDENDUM_003):                            # the real release text, when mounted
        assert F.sha256_file(F.ADDENDUM_003) == F.ADDENDUM_003_SHA


def test_a8_init_run_003_exclusion_inodes_and_chain(tmp_path):
    src, rd2 = _a8_run002(tmp_path)
    add, sha = _a8_addendum(tmp_path)
    before1, before2 = _tree(src), _tree(rd2)
    rd3 = os.path.join(str(tmp_path), 'run_003')
    lin = F.init_run_003(rd2, rd3, add, addendum_sha=sha)
    assert _tree(src) == before1 and _tree(rd2) == before2         # run_001 and run_002 untouched
    assert sorted(e['path'] for e in lin['excluded']) == A8_EXCLUDED
    assert lin['excluded_dirs'] == ['jobs/m4_ranks_gate'] and lin['excluded_patterns'] == list(A8_EXCLUDE)
    for r in A8_EXCLUDED:
        assert not os.path.lexists(os.path.join(rd3, r))
    files = {e['path']: e for e in lin['files']}
    assert set(files) == {r for r in before2 if before2[r] and r not in A8_EXCLUDED}
    for r in ('m3_matrix_gates.json', 'jobs/m3_gates/job.sh', 'c2_metric_suite/metrics_virchow2_fp32_sampler.json',
              'queue_probe/w4096_30_r002/wave.json', 'import_closure/v2m_m4_ranks_sampler.8_0.json',
              'jobs/m5_tie_hosts/job.sh', 'RUN_002_EXPECTED_NEW.json', F.LINEAGE_RUN_002):
        assert r in files and files[r]['mode'] == 'hardlink'      # carried over (incl. the run_002 lineage record)
    for r, e in files.items():
        assert e['sha256_source'] == e['sha256_dest'] == F.sha256_file(os.path.join(rd3, r)) and e['equal']
        if r in A4_COPIED:                                         # append-only: content copy, new inode
            assert e['mode'] == 'copy' and e['append_only']
            assert not os.path.samefile(os.path.join(rd2, r), os.path.join(rd3, r))
        else:
            assert os.path.samefile(os.path.join(rd2, r), os.path.join(rd3, r))
    assert lin['schema'] == F.LINEAGE_RUN_003_SCHEMA and lin['source_dir'] == rd2 and lin['dest_dir'] == rd3
    assert lin['addendum'] == {'path': add, 'sha256': sha}
    assert lin['run_002_failed'] == {'path': os.path.join(rd2, 'FAILED'),
                                     'sha256': F.sha256_file(os.path.join(rd2, 'FAILED'))}
    assert lin['run_002_lineage'] == {'path': os.path.join(rd2, F.LINEAGE_RUN_002),
                                      'sha256': F.sha256_file(os.path.join(rd2, F.LINEAGE_RUN_002)),
                                      'schema': F.LINEAGE_RUN_002_SCHEMA}
    assert 'exclude_override' not in lin and 'amendment' not in lin and 'run_001_failed' not in lin
    assert json.load(open(os.path.join(rd3, F.LINEAGE_RUN_003))) == lin
    assert not os.path.lexists(os.path.join(rd2, F.LINEAGE_RUN_003))
    # chain: run_003 -> run_002 lineage -> run_001
    l2 = json.load(open(lin['run_002_lineage']['path']))
    assert l2['dest_dir'] == rd2 and l2['source_dir'] == src
    open(os.path.join(rd3, 'journal.jsonl'), 'a').write('{"event": "submit", "step": "m4_ranks_gate", "run": 3}\n')
    assert _tree(rd2) == before2


def test_a8_init_run_002_lineage_record_unchanged_by_generalisation(tmp_path):
    """init_run_002 is now a wrapper of init_run; its record keeps the AMENDMENT_004 keys exactly."""
    src, dst, amend, sha, lin = _a4_world(tmp_path)
    assert list(lin) == ['schema', 'source_dir', 'dest_dir', 'amendment', 'run_001_failed', 'excluded',
                         'excluded_patterns', 'excluded_dirs', 'append_only_copied_names', 'directories', 'n_files',
                         'n_hardlinked', 'n_copied', 'files', 'note']
    assert lin['note'] == F.RUN_002_NOTE and lin['excluded_patterns'] == list(F.INIT_RUN_002_EXCLUDE)
    assert 'addendum' not in lin and 'run_002_failed' not in lin


def test_a8_init_run_003_refusals(tmp_path):
    add, sha = _a8_addendum(tmp_path)
    src, rd2 = _a8_run002(tmp_path / 'a', failed=False)
    before = _tree(rd2)
    d = os.path.join(str(tmp_path), 'r3_nofail')
    with pytest.raises(F.FidelityError, match='FAILED'):           # nothing to retry
        F.init_run_003(rd2, d, add, addendum_sha=sha)
    _put(rd2, 'FAILED', '{}')
    before = _tree(rd2)
    for bad in (('FAILED', 'journal.jsonl'), (F.LINEAGE_RUN_002,), ('*.json',), ('/abs',), ('a/../b',), ()):
        with pytest.raises(F.FidelityError, match='exclude'):
            F.init_run_003(rd2, d, add, addendum_sha=sha, exclude=bad)
    with pytest.raises(F.FidelityError, match='M0.3'):             # addendum binding checked before any write
        F.init_run_003(rd2, d, add, addendum_sha='0' * 64)
    assert not os.path.lexists(d)
    os.makedirs(d)
    with pytest.raises(F.FidelityError, match='already exists'):
        F.init_run_003(rd2, d, add, addendum_sha=sha)
    assert os.listdir(d) == []
    lp = os.path.join(rd2, F.LINEAGE_RUN_002)                      # no / wrong-schema source lineage record
    keep = os.path.join(str(tmp_path), 'keep.json')
    os.rename(lp, keep)
    d2 = os.path.join(str(tmp_path), 'r3_nolin')
    with pytest.raises(F.FidelityError, match='no lineage record'):
        F.init_run_003(rd2, d2, add, addendum_sha=sha)
    rec = json.load(open(keep))
    json.dump(dict(rec, schema=F.LINEAGE_RUN_003_SCHEMA), open(lp, 'w'))
    with pytest.raises(F.FidelityError, match='schema'):
        F.init_run_003(rd2, d2, add, addendum_sha=sha)
    os.remove(lp)
    os.rename(keep, lp)
    assert not os.path.lexists(d2)
    assert _tree(rd2) == before
    assert F.main(['init-run', '--source', rd2, '--dest', d2, '--addendum', add]) == 1   # sha != ADDENDUM_003_SHA
    assert not os.path.lexists(d2)


def test_a8_init_run_003_extra_exclude_recorded(tmp_path):
    src, rd2, rd3, add, sha, lin = _a8_world(tmp_path, exclude=A8_EXCLUDE + ('jobs/m5_tie_hosts/**',))
    assert lin['exclude_override'] == {'default': list(A8_EXCLUDE), 'used': list(A8_EXCLUDE) + ['jobs/m5_tie_hosts/**']}
    assert not os.path.lexists(os.path.join(rd3, 'jobs', 'm5_tie_hosts'))
    assert 'jobs/m5_tie_hosts' in lin['excluded_dirs']
    assert F.check_lineage(rd3, F.LINEAGE_RUN_003)['pass']


def test_a8_lineage_check_run_003(tmp_path):
    src, rd2, rd3, add, sha, lin = _a8_world(tmp_path)
    argv = ['lineage-check', '--stage-dir', rd3, '--lineage-file', F.LINEAGE_RUN_003]
    assert F.main(argv) == 0
    lc = F.check_lineage(rd3, F.LINEAGE_RUN_003)
    assert lc['pass'] and lc['n_unlisted'] == 0 and lc['lineage_file'] == F.LINEAGE_RUN_003
    assert lc['n_hashed'] == 2 * lin['n_files'] + len(A8_EXCLUDED)
    # the carried run_002 lineage file is an ordinary listed (hashed) file; only the stage's own record is exempt
    assert F.LINEAGE_RUN_002 in {e['path'] for e in lin['files']}
    _put(rd3, 'sub/' + F.LINEAGE_RUN_003)
    assert _roles(F.check_lineage(rd3, F.LINEAGE_RUN_003)) == [('unlisted_file', 'sub/' + F.LINEAGE_RUN_003)]
    os.remove(os.path.join(rd3, 'sub', F.LINEAGE_RUN_003))
    # the retried m4_ranks_gate regenerates excluded names: strict fails, relaxed passes; a new probe wave and
    # the allowlist file need the allowlist
    for rel in ('m4_ranks_resolution.json', 'jobs/m4_ranks_gate/job.sh', 'lsf/v2m_m4_ranks_gate.444_0.out',
                'host_sidecar/v2m_m4_ranks_gate.444_0.json', 'import_closure/v2m_m4_ranks_gate.444_0.json'):
        _put(rd3, rel, f'run_003 {rel}')
    _put(rd3, 'queue_probe/w4096_30_r003/wave.json')
    _put(rd3, 'RUN_003_EXPECTED_NEW.json')
    open(os.path.join(rd3, 'journal.jsonl'), 'a').write('{"event": "submit", "step": "m4_ranks_gate"}\n')
    lc = F.check_lineage(rd3, F.LINEAGE_RUN_003)
    assert {m['role'] for m in lc['mismatches']} == {'dest_excluded_present', 'unlisted_file'}
    lc = F.check_lineage(rd3, F.LINEAGE_RUN_003, excluded_may_be_regenerated=True)
    assert _roles(lc) == [('unlisted_file', 'RUN_003_EXPECTED_NEW.json'),
                          ('unlisted_file', 'queue_probe/w4096_30_r003/wave.json')]
    al = _allow(tmp_path, ['m4_ranks_resolution.json', 'queue_probe/*_r003/**', 'RUN_003_EXPECTED_NEW.json'])
    lc = F.check_lineage(rd3, F.LINEAGE_RUN_003, excluded_may_be_regenerated=True, expected_new=al)
    assert lc['pass'], lc['mismatches']
    assert lc['n_allowed_new'] == 2 and lc['append_only_appended'][0]['path'] == 'journal.jsonl'
    assert F.main(argv + ['--excluded-may-be-regenerated', '--expected-new', al]) == 0
    # never a carried-over run_002 file at an excluded name (run_002's own FAILED/m4_ranks_resolution bytes)
    os.remove(os.path.join(rd3, 'm4_ranks_resolution.json'))
    os.link(os.path.join(rd2, 'm4_ranks_resolution.json'), os.path.join(rd3, 'm4_ranks_resolution.json'))
    lc = F.check_lineage(rd3, F.LINEAGE_RUN_003, excluded_may_be_regenerated=True, expected_new=al)
    assert _roles(lc) == [('dest_excluded_carried_over', 'm4_ranks_resolution.json')]


def test_a8_lineage_check_source_changes_detected(tmp_path):
    src, rd2, rd3, add, sha, lin = _a8_world(tmp_path)
    open(os.path.join(rd2, 'FAILED'), 'a').write(' ')                       # run_002 FAILED changed
    open(os.path.join(rd2, 'monitor.log'), 'a').write('x')                  # append-only source changed
    lc = F.check_lineage(rd3, F.LINEAGE_RUN_003)
    assert _roles(lc) == [('source', 'monitor.log'), ('source_excluded', 'FAILED')]


def test_a8_lineage_check_name_and_dest_dir_roles(tmp_path):
    src, rd2, rd3, add, sha, lin = _a8_world(tmp_path)
    # checking run_003 against the carried run_002 record: its dest_dir is run_002, not run_003
    lc = F.check_lineage(rd3, F.LINEAGE_RUN_002)
    assert ('lineage_dest_dir_mismatch', F.LINEAGE_RUN_002) in _roles(lc) and not lc['pass']
    # a run_003 record under another name
    open(os.path.join(rd3, 'other_lineage.json'), 'w').write(open(os.path.join(rd3, F.LINEAGE_RUN_003)).read())
    lc = F.check_lineage(rd3, 'other_lineage.json')
    assert ('lineage_name_schema_mismatch', 'other_lineage.json') in _roles(lc)
    os.remove(os.path.join(rd3, 'other_lineage.json'))
    # the run_002 record still checks run_002 (dest_dir = run_002)
    lc = F.check_lineage(rd2, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True)
    assert not any(m['role'].startswith('lineage_') for m in lc['mismatches'])
    # a run_003 record without excluded_patterns is unreadable (no run_002 fallback)
    lp = os.path.join(rd3, F.LINEAGE_RUN_003)
    rec = json.load(open(lp))
    del rec['excluded_patterns']
    os.chmod(lp, 0o644)
    json.dump(rec, open(lp, 'w'))
    with pytest.raises(F.FidelityError, match='unreadable'):
        F.check_lineage(rd3, F.LINEAGE_RUN_003)


def test_a8_expected_new_accepts_any_run_lineage(tmp_path):
    src, rd2, rd3, add, sha, lin = _a8_world(tmp_path)
    _put(rd3, 'c2_metric_suite/x.json')
    al = _allow(tmp_path, ['c2_metric_suite/x.json'])
    assert F.check_lineage(rd3, F.LINEAGE_RUN_003, expected_new=al)['pass']
    assert F.check_lineage(rd2, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True,
                           expected_new=_allow(tmp_path, ['**'], name='all.json'))['pass'] is False


def test_a8_run_002_check_after_run_003_reports_shared_regenerations(tmp_path):
    """implementation_008 Issue E (reported, not decided): init-run hardlinks run_002's regenerated m3 outputs into
    run_003, so they get st_nlink 2 and a LATER re-run of run_002's relaxed lineage-check reports them under the
    ADDENDUM_002 F2 'any second link' rule. Earlier run_002 check results are unaffected; nothing in the package
    re-runs it."""
    src, rd2 = _a8_run002(tmp_path)
    assert F.check_lineage(rd2, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True,
                           expected_new=_allow(tmp_path, ['**'], name='all.json'))['pass']
    add, sha = _a8_addendum(tmp_path)
    F.init_run_003(rd2, os.path.join(str(tmp_path), 'run_003'), add, addendum_sha=sha)
    lc = F.check_lineage(rd2, F.LINEAGE_RUN_002, excluded_may_be_regenerated=True,
                         expected_new=_allow(tmp_path, ['**'], name='all.json'))
    assert {m['role'] for m in lc['mismatches']} == {'dest_excluded_carried_over'}
    assert all(m['actual'].startswith('st_nlink=2 ') for m in lc['mismatches'])
    assert sorted(m['path'] for m in lc['mismatches']) == [
        'host_sidecar/v2m_m3_gates.777_0.json', 'jobs/m3_gates/job.sh', 'jobs/m3_gates/launch.sha256',
        'jobs/m3_gates/readback_777.json', 'jobs/m3_gates/step.json', 'lsf/v2m_m3_gates.777_0.err',
        'lsf/v2m_m3_gates.777_0.out', 'm3_matrix_gates.json']


def test_a8_cli_init_run_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(F, 'cmd_init_run', lambda a: seen.__setitem__('a', a) or 0)
    assert F.main(['init-run']) == 0
    a = seen['a']
    assert (a.source, a.dest, a.addendum, a.extra_exclude, a.copy) == (F.RUN_DIR_002, F.RUN_DIR_003,
                                                                        F.ADDENDUM_003, [], False)
    assert F.main(['init-run', '--extra-exclude', 'jobs/m5_tie_hosts/**', 'x/**']) == 0
    assert seen['a'].extra_exclude == ['jobs/m5_tie_hosts/**', 'x/**']


def test_a8_cmd_init_run_extra_exclude_appended(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(F, 'init_run_003', lambda *a, **k: seen.update(a=a, k=k) or {
        'dest_dir': 'd', 'n_files': 0, 'n_hardlinked': 0, 'n_copied': 0, 'files': [], 'excluded': [],
        'excluded_dirs': [], 'excluded_patterns': []})
    assert F.main(['init-run', '--source', 's', '--dest', 'd', '--addendum', 'a',
                   '--extra-exclude', 'FAILED', 'jobs/m5_tie_hosts/**']) == 0
    assert seen['a'] == ('s', 'd', 'a') and seen['k']['exclude'] == A8_EXCLUDE + ('jobs/m5_tie_hosts/**',)
    assert seen['k']['copy'] is False
