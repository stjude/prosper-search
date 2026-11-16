"""Synthetic-fixture tests for scripts/r2_1_virchow2_fp32_main_rr5_adapter.py (V2-main, R2.1).

The synthetic RR-5 world of the frozen tests/test_r2_1_main_grid_plurality.py (read-only import) is extended with
four `virchow2_fp32/<m>` matrices (perturbed copies of the synthetic virchow2 matrices), an evaluator attempt
(summary/query_metrics/draws/populations/estimates) and a 42-style pins file. RR-5 itself runs through the
adapter's hash-checked namespace: Run A, Run B (option (ii)) and B2. No real data are read. Run with an explicit
--basetemp outside /tmp, `-p no:cacheprovider` and PYTHONDONTWRITEBYTECODE=1.
"""
import gzip
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
ADAPTER = os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_rr5_adapter.py')
FROZEN_TESTS = os.path.join(REPO, 'tests', 'test_r2_1_main_grid_plurality.py')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load('v2main_rr5_adapter_under_test', ADAPTER)
T = _load('rr5_frozen_test_world', FROZEN_TESTS)
METHODS = ('median_pooling', 'ot_cosine', 'sampler', 'yottixel')


# ----------------------------------------------------------------------------------------------
# world builders (also used by test_r2_1_virchow2_fp32_main_compare.py)
# ----------------------------------------------------------------------------------------------

def macro_points(w, cid, k):
    """Independent macro hit@k / precision@k: mean over groups of the within-group mean."""
    p = w['pred'][cid]['canonical'][k]
    y = w['y']
    G = len(T.GROUPS)
    hit = np.mean([p['hit'][y == g].mean() for g in range(G)])
    prec = np.mean([(p['c'][y == g] / k).mean() for g in range(G)])
    return float(hit), float(prec)


def add_fp32(w, run_dir, seed=11, scale=0.25):
    rng = np.random.default_rng(seed)
    ids, pat, y = w['ids'], w['pat'], w['y']
    for m in METHODS:
        src = w['mats'][f'virchow2/{m}']
        V = src.to_numpy()
        sent = V == T.M.SENTINEL
        E = rng.normal(0, scale, size=V.shape)
        E = (E + E.T) / 2.0
        V2 = np.abs(V + E)
        if m == 'yottixel':
            V2 = np.round(V2 * 1.5) / 1.5
        V2[sent] = T.M.SENTINEL
        df = pd.DataFrame(V2, index=src.index, columns=src.columns)
        cid = f'virchow2_fp32/{m}'
        p = os.path.join(run_dir, 'similarity_matrix',
                         f'combined_similarity_matrix_virchow2_fp32_{m}_no_same_patient.pkl')
        T.wbytes(p, pickle.dumps(df))
        Dp = df.loc[ids, ids].to_numpy().copy()
        Dp[pat[:, None] == pat[None, :]] = np.inf
        w['D'][cid] = Dp
        w['mats'][cid] = df
        w['pred'][cid] = {o: T.ref_predict(Dp, pat, y, reverse=(o == 'reversed')) for o in ('canonical', 'reversed')}
    ev = os.path.join(run_dir, 'eval_attempt_001')
    pth = w['spec']['paths']
    T.wbytes(os.path.join(ev, 'summary.json'),
             json.dumps({'policy': {'patient_order': w['P']}, 'encoder': 'virchow2_fp32'}).encode())
    T.wbytes(os.path.join(ev, 'query_metrics.csv.gz'), T.qm_frame(w, 'virchow2_fp32'))
    shutil.copyfile(pth['draws'], os.path.join(ev, 'patient_draws.npy'))
    shutil.copyfile(pth['population'], os.path.join(ev, 'populations.csv'))
    rows = []
    for m in METHODS:
        for k in T.KS:
            h, pr = macro_points(w, f'virchow2_fp32/{m}', k)
            for ep, v in (('hit', h), ('precision', pr)):
                rows.append(dict(column=f'{m}_{ep}_{k}', row_kind='estimate', method=m, arm_a='', arm_b='',
                                 endpoint=ep, k=k, level='macro', group='', point=repr(v)))
                rows.append(dict(column=f'{m}_{ep}_{k}_g', row_kind='estimate', method=m, arm_a='', arm_b='',
                                 endpoint=ep, k=k, level='group', group='ALPHA', point='0.123'))
    pd.DataFrame(rows).to_csv(os.path.join(ev, 'estimates.csv'), index=False)
    return ev


def build_v2_world(root, seed=7):
    """Synthetic world + run_001 (38-style baseline) + fp32 run dir with eval attempt and pins file."""
    w = T.make_world(root, seed=seed)
    rc, run001 = T.run_cli(w, 'run001')
    assert rc == 0
    run_dir = os.path.join(root, 'v2main', 'run_001')
    ev = add_fp32(w, run_dir)
    ns = A.load_rr5()
    A.add_fp32_display(ns)
    pins_path = os.path.join(run_dir, 'pins_v2main_001.json')
    tmp = A.build_full_spec(ns, w['spec'], A.fp32_matrix_paths(run_dir, METHODS), ev, pins_path, 'pending')
    pins = A.build_pins(open(w['spec']['paths']['pins_file'], 'rb').read(), tmp['cells'], ev,
                        {'head': 'synthetic', 'files': {}})
    A.write_new(pins_path, pins)
    return {'w': w, 'ns': ns, 'run001': run001, 'run_dir': run_dir, 'eval': ev, 'pins': pins_path}


def run_a(v):
    return A.do_run_a(v['ns'], v['w']['spec'], v['run_dir'], v['run001'], T.R_REPS, v['pins'])


def signoff_binding(v):
    """Synthetic feasibility report + analyst sign-off naming its sha256 and option (ii) (F10)."""
    d = os.path.dirname(v['run_dir'])
    feas = os.path.join(d, 'RUN_B_FEASIBILITY.md')
    if not os.path.exists(feas):
        with open(feas, 'w') as f:
            f.write('synthetic feasibility report: option (ii)\n')
    so = os.path.join(d, 'RUN_B_SIGNOFF.md')
    if not os.path.exists(so):
        with open(so, 'w') as f:
            f.write(f'Lead signs RUN_B_FEASIBILITY.md sha256 {A.sha256_file(feas)}: option (ii) adopted\n')
    return A.check_signoff(so, feas)


def run_b(v):
    return A.do_run_b(A.RR5_PATH, A.RR5_SHA, v['w']['spec'], v['run_dir'], v['run001'], T.R_REPS, v['pins'],
                      signoff_binding(v))


@pytest.fixture(scope='module')
def v2(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('v2world'))
    v = build_v2_world(root)
    v['run_a'] = run_a(v)
    v['run_b'] = run_b(v)
    return v


# ----------------------------------------------------------------------------------------------
# namespace / spec / pins
# ----------------------------------------------------------------------------------------------

def test_load_rr5_hash_checked_fresh_namespaces():
    a = A.load_rr5()
    b = A.load_rr5()
    assert a is not b and a.ENC_DISPLAY is not b.ENC_DISPLAY
    assert a.__rr5_sha256__ == A.RR5_SHA
    A.add_fp32_display(a)
    assert a.ENC_DISPLAY['virchow2_fp32'] == 'Virchow2' and 'virchow2_fp32' not in b.ENC_DISPLAY
    assert a.XLSX_EMB['Virchow2'] == 'virchow2'
    with pytest.raises(A.AdapterError):
        A.add_fp32_display(a)
    with pytest.raises(A.AdapterError):
        A.load_rr5(expected_sha='0' * 64)


def test_full_spec_and_rb_spec_key_lists():
    ns = A.load_rr5()
    A.add_fp32_display(ns)
    base = ns.contract_spec('/r', '/c', '/o')
    # build_full_spec hashes the fp32 files: use this adapter source as a stand-in file for the shape test
    paths = {m: ADAPTER for m in METHODS}
    ev = os.path.dirname(ADAPTER)
    orig = A.sha256_file
    try:
        A.sha256_file = lambda p: 'f' * 64
        fs = A.build_full_spec(ns, base, paths, ev, '/o/pins_v2main_001.json', 'a' * 64)
    finally:
        A.sha256_file = orig
    assert len(fs['cells']) == 42 and fs['cells'][:38] == base['cells']
    c = fs['cells'][38]
    assert c == dict(cell_id='virchow2_fp32/median_pooling', enc='virchow2_fp32', method='median_pooling',
                     display_encoder='Virchow2', aggregation=ns.AGG_DISPLAY['median_pooling'],
                     lineage='virchow2_fp32_replacement', submitted_lineage=False, reconcile=False, path=ADAPTER,
                     sha256='f' * 64, hist_prefix=None, xlsx_key=None)
    assert 'published_counterpart' not in c
    assert fs['qm_encoders'] == ['uni2', 'virchow2', 'h_optimus_1', 'virchow2_fp32']
    assert fs['n_hist_rows'] == 132 and fs['expected_partition'] == base['expected_partition']
    rb, diff = A.make_rb_spec(fs)
    assert [x['leaf'] for x in diff] == ['contested_encoders[6]', 'fig2c.pipelines[2]'] or \
        sorted(x['new'] for x in diff) == ['virchow2_fp32', 'virchow2_fp32/yottixel']
    assert len(diff) == 2
    assert rb['contested_encoders'].index('virchow2_fp32') == base['contested_encoders'].index('virchow2')
    assert rb['fig2c']['pipelines'] == ['titan/titan_repaired', 'uni2/ot_cosine', 'virchow2_fp32/yottixel']
    assert rb['cells'] == fs['cells'] and rb['qm_encoders'] == fs['qm_encoders']
    assert rb['fig2c']['xlsx_pipelines'] == fs['fig2c']['xlsx_pipelines']
    with pytest.raises(A.AdapterError):
        A.make_rb_spec(rb)   # already substituted


def test_alias_wrapper_identity_and_counts():
    ns = A.load_rr5()
    rows = {(f'virchow2/{m}', k): {'Overall Metrics': {'Macro F1': 0.5}} for m in METHODS for k in ns.K_VALUES}
    ns.parse_xlsx = lambda spec, inp, popd, hist: {'rows': rows}
    rec = {}
    A.install_alias_wrapper(ns, rec)
    spec = {'cells': [{'cell_id': f'virchow2_fp32/{m}', 'enc': 'virchow2_fp32', 'method': m} for m in METHODS]}
    xl = ns.parse_xlsx(spec, None, None, None)
    assert rec['alias_count'] == 16 and rec['calls'] == 1 and len(rec['aliases']) == 16
    for m in METHODS:
        for k in ns.K_VALUES:
            assert xl['rows'][(f'virchow2_fp32/{m}', k)] is xl['rows'][(f'virchow2/{m}', k)]
    with pytest.raises(A.AdapterError):
        A.install_alias_wrapper(ns, {})
    ns2 = A.load_rr5()
    ns2.parse_xlsx = lambda spec, inp, popd, hist: {'rows': {}}
    rec2 = {}
    A.install_alias_wrapper(ns2, rec2)
    ns2.parse_xlsx({'cells': []}, None, None, None)
    assert rec2['alias_count'] == 0


def test_assert_writable_dir_rejects_frozen_and_tmp(tmp_path):
    for bad in ('/tmp/x', os.path.join(str(tmp_path), 'r2_1_main_grid_plurality_20261012', 'x'),
                os.path.join(str(tmp_path), 'contract_candidate_002')):
        with pytest.raises(A.AdapterError):
            A.assert_writable_dir(bad)
    d = tmp_path / 'full'
    d.mkdir()
    (d / 'f').write_text('x')
    with pytest.raises(A.AdapterError):
        A.assert_writable_dir(str(d))
    with pytest.raises(A.AdapterError):
        A.write_new(str(d / 'f'), b'y')


def test_pre_run_asserts_negative(tmp_path):
    v = build_v2_world(str(tmp_path))
    ns = v['ns']
    spec, pre = A.prepare_full_spec(ns, v['w']['spec'], v['run_dir'], v['pins'])
    assert pre['n_matrices'] == len(spec['cells']) and pre['n_fp32'] == 4
    pf = json.loads(open(v['pins']).read())
    del pf['matrices']['virchow2_fp32/sampler']
    with pytest.raises(A.AdapterError):
        A.pre_run_asserts(spec, A.canonical_json(pf))
    pf = json.loads(open(v['pins']).read())
    pf['matrices']['virchow2_fp32/sampler']['sha256'] = '0' * 64
    with pytest.raises(A.AdapterError):
        A.pre_run_asserts(spec, A.canonical_json(pf))
    with pytest.raises(A.AdapterError):    # pins sha must equal spec pins.pins_file
        A.pre_run_asserts(spec, open(v['pins'], 'rb').read() + b' ')


# ----------------------------------------------------------------------------------------------
# Run A / Run B on the synthetic world
# ----------------------------------------------------------------------------------------------

def test_run_a_complete_and_bit_identical(v2):
    spec, rep, xc = v2['run_a']
    out = os.path.join(v2['run_dir'], 'rr5_additive_001')
    st = json.load(open(os.path.join(out, 'COMPLETE.json')))
    assert st['status'] == 'COMPLETE' and len(st['gates_passed']) == 13 and st['spec_is_contract'] is False
    assert rep['pass'] and not rep['failures']
    assert all(rep['a4_whole_file'].values())
    assert all(x['identical_after_removing_fp32_rows'] for x in rep['a4_filtered'].values())
    assert rep['a4_filtered']['fig2b_matrix.csv']['fp32_rows_removed'] > 0
    assert rep['reconciliation']['hit_gate.virchow2_fp32'] == 'added'
    assert set(rep['fp32_order_sensitive_10']) == {f'virchow2_fp32/{m}' for m in METHODS}
    assert xc['pass'] and xc['n_compared'] == 32
    assert max(r['abs_diff'] for r in xc['rows']) <= 1e-12
    # hit_gate compared every fp32 per-query row
    hg = json.load(open(os.path.join(out, 'reconciliation.json')))['hit_gate']['virchow2_fp32']
    assert hg == {'rows_checked': 4 * 4 * v2['w']['n'], 'hit_mismatches': 0, 'precision_mismatches': 0}
    assert rep['hit_gate_fp32_rows_expected'] == 4 * 4 * v2['w']['n']
    tc = pd.read_csv(os.path.join(out, 'tie_census.csv'))
    assert set(tc['cell_id']) >= {f'virchow2_fp32/{m}' for m in METHODS}


def test_run_b_option_ii(v2):
    rep = v2['run_b']
    rb = os.path.join(v2['run_dir'], 'rr5_replacement_001')
    assert rep['pass'] and rep['alias_count'] == 16 and rep['b2_alias_count'] == 0
    assert all(rep['b1_whole_file_identity'].values()) and all(rep['b2_faithfulness'].values())
    assert len(rep['b3_gates_passed']) == 13
    assert rep['oracle_winner_count_reproduces_submitted']['note'] == A.A8_NOTE
    a10 = rep['a10_row_drop']
    assert a10['table_s3_revised.csv']['dropped'] > 0 and a10['winners.csv']['dropped'] > 0
    assert a10['fig2a_panel.csv']['dropped'] == 0 and a10['aggregation_contrasts.csv']['dropped'] == 0
    rbs = json.load(open(os.path.join(rb, 'rb_spec.json')))
    assert rbs['option'] == '(ii)' and len(rbs['rb_spec_vs_full_spec']) == 2
    assert rbs['run_a_sha256sums_sha256'] == A.sha256_file(
        os.path.join(v2['run_dir'], 'rr5_additive_001', 'SHA256SUMS.txt'))
    f2c = pd.read_csv(os.path.join(rb, 'fig2c_groups.csv'), keep_default_na=False)
    fp = f2c[f2c['pipeline'] == 'virchow2_fp32/yottixel']
    assert len(fp) and (fp['submitted_from'] == A.FIG2C_FP32_LABEL).all()
    assert not f2c['pipeline'].str.startswith('virchow2/').any()
    f2b = pd.read_csv(os.path.join(rb, 'fig2b_matrix.csv'), keep_default_na=False)
    assert 'published_8' not in set(f2b['scope']) and 'published_7+virchow2_fp32' in set(f2b['scope'])
    t3 = pd.read_csv(os.path.join(rb, 'table_s3_revised.csv'), keep_default_na=False, dtype=str)
    fpr = t3[t3['cell_id'].str.startswith('virchow2_fp32/')]
    assert (fpr['submitted_source'] != '').all()
    assert (t3.loc[~t3['cell_id'].str.startswith('virchow2_fp32/'), 'submitted_source'] == '').all()
    ta = pd.read_csv(os.path.join(v2['run_dir'], 'rr5_additive_001', 'table_s3_revised.csv'),
                     keep_default_na=False, dtype=str)
    pub = ta[ta['cell_id'] == 'virchow2/yottixel'].reset_index(drop=True)
    mine = fpr[fpr['cell_id'] == 'virchow2_fp32/yottixel'].reset_index(drop=True)
    for c in A.SUBMITTED_FILL_COLS:
        assert list(mine[c]) == list(pub[c])
    # every non-table_s3 retained line is a raw Run B line
    raw = os.path.join(rb, 'raw')
    for name in ('hit_precision.csv', 'per_class.csv', 'winners.csv', 'fig2a_panel.csv'):
        fin = A.split_lines(A.read_product(rb, name))[1]
        rl = set(A.split_lines(A.read_product(raw, name))[1])
        assert set(fin) <= rl
    assert A.u4_assert(rb) == []
    assert not A.u4_assert(rb)


def test_run_b_requires_complete_run_a(tmp_path):
    v = build_v2_world(str(tmp_path))
    with pytest.raises((A.AdapterError, OSError)):
        run_b(v)


def test_bit_identity_detects_changed_row(v2, tmp_path):
    bad = str(tmp_path / 'run001_bad')
    shutil.copytree(v2['run001'], bad)
    p = os.path.join(bad, 'hit_precision.csv')
    lines = open(p, 'rb').read().split(b'\n')
    lines[1] = lines[1].replace(b'0.', b'1.', 1)
    open(p, 'wb').write(b'\n'.join(lines))
    spec = v2['run_a'][0]
    orig = [c['cell_id'] for c in spec['cells'] if c['enc'] != 'virchow2_fp32']
    fp = [c['cell_id'] for c in spec['cells'] if c['enc'] == 'virchow2_fp32']
    rep = A.bit_identity_report(os.path.join(v2['run_dir'], 'rr5_additive_001'), bad, orig, fp)
    assert not rep['pass'] and any('hit_precision.csv' in f for f in rep['failures'])


def test_evaluator_crosscheck_detects_1e9(v2, tmp_path):
    est = pd.read_csv(os.path.join(v2['eval'], 'estimates.csv'), dtype=str)
    i = est.index[(est['level'] == 'macro')][0]
    est.loc[i, 'point'] = repr(float(est.loc[i, 'point']) + 1e-9)
    p = str(tmp_path / 'estimates.csv')
    est.to_csv(p, index=False)
    ns = A.load_rr5()
    A.add_fp32_display(ns)
    spec = v2['run_a'][0]
    fp = [c['cell_id'] for c in spec['cells'] if c['enc'] == 'virchow2_fp32']
    stats, _ = A.recompute_cell_stats(ns, spec, fp, str(tmp_path / 'bind'), T.R_REPS)
    xc = A.evaluator_crosscheck(ns, stats, p, os.path.join(v2['run_dir'], 'rr5_additive_001'), METHODS, T.KS)
    assert not xc['pass'] and len(xc['failures']) == 1


def test_a10_rejects_unexpected_virchow2_row(v2, tmp_path):
    raw = str(tmp_path / 'raw')
    shutil.copytree(os.path.join(v2['run_dir'], 'rr5_replacement_001', 'raw'), raw)
    p = os.path.join(raw, 'fig2a_panel.csv')
    head, lines = A.split_lines(open(p, 'rb').read())
    f = A.parse_fields(lines[0])
    hf = A.parse_fields(head)
    f[hf.index('cell_id')] = 'virchow2/sampler'
    open(p, 'wb').write(A.join_lines(head, [A.format_fields(f)] + lines[1:]))
    _, counts, bad = A.a10_drop(raw, os.path.join(v2['run_dir'], 'rr5_additive_001'))
    assert any('fig2a_panel.csv' in b for b in bad)
    assert counts['fig2a_panel.csv']['dropped'] == 1


def test_u4_detects_virchow2_identifier(v2, tmp_path):
    fin = str(tmp_path / 'final')
    shutil.copytree(os.path.join(v2['run_dir'], 'rr5_replacement_001'), fin,
                    ignore=shutil.ignore_patterns('raw', 'b2_faithfulness'))
    p = os.path.join(fin, 'winners.csv')
    head, lines = A.split_lines(open(p, 'rb').read())
    hf = A.parse_fields(head)
    for i, ln in enumerate(lines):
        f = A.parse_fields(ln)
        if f[hf.index('winner_cell')].startswith('virchow2_fp32/'):
            f[hf.index('winner_cell')] = f[hf.index('winner_cell')].replace('virchow2_fp32/', 'virchow2/')
            lines[i] = A.format_fields(f)
            break
    os.chmod(p, 0o644)
    open(p, 'wb').write(A.join_lines(head, lines))
    assert A.u4_assert(fin)


def test_signoff_required(tmp_path):
    feas = tmp_path / 'RUN_B_FEASIBILITY.md'
    feas.write_text('report')
    with pytest.raises(A.AdapterError):
        A.check_signoff(None, str(feas))
    s = tmp_path / 'signoff.txt'
    s.write_text('signed option (ii) for ' + '0' * 64)
    with pytest.raises(A.AdapterError):
        A.check_signoff(str(s), str(feas))
    s2 = tmp_path / 'signoff2.txt'
    s2.write_text(f'Lead signs RUN_B_FEASIBILITY.md sha256 {A.sha256_file(str(feas))}: option (ii) adopted')
    A.check_signoff(str(s2), str(feas))
    assert A.main(['run-b', '--run-dir', str(tmp_path / 'rd'), '--signoff', str(s), '--feasibility', str(feas)]) == 1


# ----------------------------------------------------------------------------------------------
# M4.2 evaluator spec / launch / job derivation
# ----------------------------------------------------------------------------------------------

def _template(tmp_path):
    src = {}
    for m in METHODS:
        for kind in ('matrix', 'ranks'):
            p = tmp_path / 'tmpl' / f'{kind}_{m}.bin'
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(f'{kind}{m}'.encode())
            src[f'{kind}_{m}'] = {'path': str(p), 'sha256': A.sha256_file(str(p)), 'size': p.stat().st_size}
    src['metadata'] = {'path': '/x/meta.csv', 'sha256': '1' * 64, 'size': 3}
    t = {'run_id': 'old', 'sources': src, 'policy': {'n_reps': 5000}}
    return (json.dumps(t, indent=2, sort_keys=True) + '\n').encode(), t


def test_eval_spec_25_leaves(tmp_path):
    tb, t = _template(tmp_path)
    mats, ranks = {}, {}
    for m in METHODS:
        for kind, d in (('matrix', mats), ('ranks', ranks)):
            p = tmp_path / 'new' / f'{kind}_{m}.bin'
            p.parent.mkdir(exist_ok=True)
            # sampler files keep the template size: their size leaves are listed with value_changed False
            p.write_bytes((f'NEW{kind}{m}' if m != 'sampler' else f'{kind}{m}'.upper()).encode())
            d[m] = str(p)
    out, diff = A.eval_spec(tb, mats, ranks)
    assert diff['n_leaves'] == 25 and not diff['unexpected_changes']
    assert diff['n_value_changed'] == 23
    assert {x['leaf'] for x in diff['leaves'] if not x['value_changed']} == {
        'sources.matrix_sampler.size', 'sources.ranks_sampler.size'}
    s = json.loads(out)
    assert s['policy'] == t['policy'] and s['sources']['metadata'] == t['sources']['metadata']
    assert s['run_id'] == A.EVAL_RUN_ID
    with pytest.raises(A.AdapterError):
        A.eval_spec(tb + b' ', mats, ranks)    # non-canonical template


def test_eval_job_line_substitution():
    tmpl = '\n'.join(['#!/bin/bash', '#BSUB -J old', '#BSUB -q priority', '#BSUB -oo /a/o', '#BSUB -eo /a/e',
                      'export PROSPER_OUT=/a', "printf '%s\\n' 'x  /a/l' | sha256sum --check --strict -",
                      'sha256sum --check --strict "/a/l"', 'test ! -e /a/att', 'python -m x \\',
                      '  --spec /a/s --spec-sha256 y \\', '  --out /a/att --out-root /a', '']).encode()
    job, diff = A.eval_job(tmpl, 'heavy_io', '/r/run_001', '/r/run_001/s.json', 's' * 64, '/r/run_001/l', 'l' * 64)
    j = job.decode()
    assert '#BSUB -q heavy_io' in j and 'export PROSPER_OUT="/r/run_001"' in j and 'python -m x \\' in j
    assert b'main_virchow2_job.lsf' in diff
    with pytest.raises(A.AdapterError):
        A.eval_job(tmpl.replace(b'#BSUB -J old\n', b''), 'q', '/r', '/s', 's', '/l', 'l')


def test_eval_launch_rebinds_only_declared_lines():
    tl = b'aa  /t/spec.json\nbb  /t/m1\ncc  /t/r1\ndd  /t/code.py\n'
    out = A.eval_launch(tl, '/t/spec.json', '/n/spec.json', 'ss',
                        {'matrix_x': {'path': '/t/m1'}, 'ranks_x': {'path': '/t/r1'}},
                        {'matrix_x': {'path': '/n/m1', 'sha256': 'm'}, 'ranks_x': {'path': '/n/r1', 'sha256': 'r'}})
    assert out == b'ss  /n/spec.json\nm  /n/m1\nr  /n/r1\ndd  /t/code.py\n'


def test_gz_like_rr5_matches_rr5_writer(v2):
    raw = A.read_product(v2['run001'], 'per_query_predictions.csv.gz')
    assert A.gz_compress_like_rr5(raw) == open(os.path.join(v2['run001'], 'per_query_predictions.csv.gz'),
                                               'rb').read()
    assert gzip.decompress(A.gz_compress_like_rr5(b'a\n')) == b'a\n'


# ----------------------------------------------------------------------------------------------
# repair candidate 002 regressions (implementation review 001: F3, F5, F8, F9, F10)
# ----------------------------------------------------------------------------------------------

def test_f3_a8_amended_mapping_both_scopes(v2):
    assert A.A8_SCOPE_MAP == {'published_8': 'published_7+virchow2_fp32',
                              'published_8_plus_h_optimus_1': 'published_7+virchow2_fp32_plus_h_optimus_1'}
    rep = v2['run_b']
    by = rep['post_call_edits']['fig2b_scope_relabelled_by_scope']
    assert set(by) == set(A.A8_SCOPE_MAP) and all(n > 0 for n in by.values())
    assert rep['post_call_edits']['fig2b_scope_relabelled'] == sum(by.values())
    rb = os.path.join(v2['run_dir'], 'rr5_replacement_001')
    raw = pd.read_csv(os.path.join(rb, 'raw', 'fig2b_matrix.csv'), keep_default_na=False, dtype=str)
    fin = pd.read_csv(os.path.join(rb, 'fig2b_matrix.csv'), keep_default_na=False, dtype=str)
    assert {'published_8', 'published_8_plus_h_optimus_1'} <= set(raw['scope'])
    assert not set(fin['scope']) & set(A.A8_SCOPE_MAP)
    for frm, to in A.A8_SCOPE_MAP.items():
        assert (fin['scope'] == to).sum() == by[frm]


def test_f5_recompute_ctx_records_w_sha(v2, tmp_path):
    ns = A.load_rr5()
    A.add_fp32_display(ns)
    spec = v2['run_a'][0]
    seen = []
    orig = ns.cell_statistics

    def spy(W, *a, **k):
        seen.append(W)
        return orig(W, *a, **k)
    ns.cell_statistics = spy
    cids = [c['cell_id'] for c in spec['cells'] if c['enc'] in ('virchow2_fp32', 'virchow2')]
    _, ctx = A.recompute_cell_stats(ns, spec, cids, str(tmp_path / 'bind'), T.R_REPS)
    assert len(seen) == len(cids) == 8 and all(w is seen[0] for w in seen)
    assert ctx['W_sha256'] == A.sha256_bytes(np.ascontiguousarray(seen[0]).tobytes())


def test_f8_cli_refuses_rr5_override(tmp_path, capsys):
    other = tmp_path / 'rr5_copy.py'
    shutil.copy(A.RR5_PATH, other)
    assert A.main(['--rr5', str(other), '--rr5-sha', A.RR5_SHA, 'internals']) == 1
    assert 'F8' in capsys.readouterr().err
    assert A.main(['--rr5-sha', '1' * 64, 'internals']) == 1
    assert 'F8' in capsys.readouterr().err
    with pytest.raises(A.AdapterError, match='F8'):
        A.require_pinned_rr5(str(other), A.RR5_SHA)
    A.require_pinned_rr5(A.RR5_PATH, A.RR5_SHA)


def test_f9_verify_sha256sums(v2, tmp_path):
    ra = os.path.join(v2['run_dir'], 'rr5_additive_001')
    ok = A.verify_sha256sums(ra)
    assert ok['lines_verified'] == len(open(os.path.join(ra, 'SHA256SUMS.txt')).read().splitlines())
    d = str(tmp_path / 'ra')
    shutil.copytree(ra, d)
    p = os.path.join(d, 'winners.csv')
    os.chmod(p, 0o644)
    open(p, 'ab').write(b'x')
    with pytest.raises(A.AdapterError, match='winners.csv: sha256 mismatch'):
        A.verify_sha256sums(d)
    d2 = str(tmp_path / 'ra2')
    shutil.copytree(ra, d2)
    s = os.path.join(d2, 'SHA256SUMS.txt')
    os.chmod(s, 0o644)
    open(s, 'a').write(f'{"0" * 64}  sub/dir.csv\n')
    with pytest.raises(A.AdapterError, match='malformed'):
        A.verify_sha256sums(d2)


def test_f9_run_b_refuses_tampered_run_a(tmp_path):
    v = build_v2_world(str(tmp_path))
    run_a(v)
    p = os.path.join(v['run_dir'], 'rr5_additive_001', 'table_s3_revised.csv')
    os.chmod(p, 0o644)
    data = open(p, 'rb').read()
    open(p, 'wb').write(data.replace(b'0.', b'1.', 1))
    with pytest.raises(A.AdapterError, match='F9'):
        run_b(v)
    assert not os.path.exists(os.path.join(v['run_dir'], 'rr5_replacement_001'))


def test_f10_signoff_bound_in_reports(v2):
    rb = os.path.join(v2['run_dir'], 'rr5_replacement_001')
    so = os.path.join(os.path.dirname(v2['run_dir']), 'RUN_B_SIGNOFF.md')
    want = {'path': os.path.abspath(so), 'sha256': A.sha256_file(so)}
    fr = json.load(open(os.path.join(rb, 'faithfulness_report.json')))
    rs = json.load(open(os.path.join(rb, 'rb_spec.json')))
    assert fr['signoff'] == want and rs['signoff'] == want
    assert fr['b4']['run_a_sha256sums_lines_verified'] == rs['run_a_sha256sums_lines_verified'] > 0


def test_f10_run_b_requires_binding_and_unchanged_signoff(tmp_path):
    v = build_v2_world(str(tmp_path))
    with pytest.raises(A.AdapterError, match='F10'):
        A.do_run_b(A.RR5_PATH, A.RR5_SHA, v['w']['spec'], v['run_dir'], v['run001'], T.R_REPS, v['pins'])
    b = signoff_binding(v)
    with open(b['path'], 'a') as f:
        f.write('edited after check\n')
    with pytest.raises(A.AdapterError, match='F10'):
        A.do_run_b(A.RR5_PATH, A.RR5_SHA, v['w']['spec'], v['run_dir'], v['run001'], T.R_REPS, v['pins'], b)
