"""Synthetic-fixture tests for scripts/r2_1_filter_rer_routeb.py (A-5, R2.1-ROUTE-B-UNI2 specification candidate 002).

No large data are read. The frozen B(c) script and the frozen vote module (code) are executed from their verified
bytes. The only real files read are run_001's contrasts.csv header line and, if present, the ten small H-Optimus-1
`_filter_summary.csv` census files (regression of the EXPECTED_NONRETURN derivation). Run with an explicit
--basetemp outside /tmp and without the cache provider.
"""
import gzip
import hashlib
import importlib.util
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, 'scripts', 'r2_1_filter_rer_routeb.py')
FROZEN = os.path.join(REPO, 'scripts', 'r2_1_filter_rer.py')
OUT_REAL = os.path.join(REPO, 'prosper_revision_output')
GROUPS = ['ACT/ACC', 'DSRCT', 'EWS', 'LIVER', 'MEL', 'MPNST', 'MRT', 'NB', 'NF', 'OS', 'RB', 'RMS', 'SYN', 'WT']


def _load():
    spec = importlib.util.spec_from_file_location('r2_1_filter_rer_routeb_under_test', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A5 = _load()


def sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def M():
    mod, info = A5.load_frozen_rer()
    assert info['sha256'] == A5.FROZEN_RER_SHA256
    return mod


# ----------------------------------------------------------------------------------------------
# Synthetic world
# ----------------------------------------------------------------------------------------------

N_SLIDES, N_PAT, N_REPS = 70, 50, 40
NONRET = {'5.0': {1003, 1011}, '100.0': {1003, 1020, 1033}}


def _population():
    rows = []
    for i in range(N_SLIDES):
        rows.append({'slide_id': 1000 + i, 'patient_id': f'P{i % N_PAT:03d}',
                     'diagnostic_group': GROUPS[(i % N_PAT) % len(GROUPS)]})
    return pd.DataFrame(rows)


def _arm_tables(pop, rng):
    """query_metrics / ranked_lists in the frozen evaluator's schema for 1 + 4 arms x 2 methods."""
    qm_rows, rl_rows = [], []
    group_of = dict(zip(pop.slide_id, pop.diagnostic_group))
    pat_of = dict(zip(pop.slide_id, pop.patient_id))
    ids = list(pop.slide_id)
    arms = [('all_tissue', None)] + [(d, b) for b in ('5', '100') for d in ('query_only', 'both_filtered')]
    for m in ('ot_cosine', 'yottixel'):
        for a, b in arms:
            nonret = set() if b is None else NONRET[f'{b}.0']
            for q in ids:
                ret = q not in nonret
                cands = [x for x in ids if pat_of[x] != pat_of[q]]
                nb = list(rng.choice(cands, size=10, replace=False))
                d = np.sort(rng.uniform(0.1, 1.0, size=10))
                if m == 'yottixel' and q % 7 == 0:
                    d[4] = d[5]  # boundary tie block (ranks 5-6) inside the stored top-10
                hit5 = float(any(group_of[x] == group_of[q] for x in nb[:5])) if ret else np.nan
                for k in (1, 3, 5, 10):
                    qm_rows.append({'method': m, 'arm': a, 'budget': b, 'k': k, 'query': q,
                                    'status': 'returned' if ret else 'empty_query',
                                    'boundary_tie': int(ret and d[4] == d[5]) if k == 5 else 0,
                                    'conditional_hit': hit5 if k == 5 else np.nan,
                                    'diagnostic_group': group_of[q], 'patient_id': pat_of[q]})
                if ret:
                    for r in range(10):
                        rl_rows.append({'method': m, 'arm': a, 'budget': b, 'query': q, 'rank': r + 1,
                                        'neighbour': nb[r], 'neighbour_group': group_of[nb[r]],
                                        'relevant': int(group_of[nb[r]] == group_of[q]), 'distance': d[r]})
    return pd.DataFrame(qm_rows), pd.DataFrame(rl_rows)


def _gz(df):
    return gzip.compress(df.to_csv(index=False).encode(), mtime=0)


@pytest.fixture()
def world(tmp_path, M):
    rng = np.random.default_rng(7)
    root = tmp_path / 'OUT' / A5.ROOT_NAME
    out_real = tmp_path / 'OUT'
    pop = _population()
    popd = out_real / 'r2_1_mosaic_cpu_20261003' / 'attempt_001'
    popd.mkdir(parents=True)
    pop.to_csv(popd / 'retrieval_population.csv', index=False)
    ev = root / 'evaluation_qf'
    ev.mkdir(parents=True)
    qm, rl = _arm_tables(pop, rng)
    (ev / 'query_metrics.csv.gz').write_bytes(_gz(qm))
    (ev / 'ranked_lists.csv.gz').write_bytes(_gz(rl))
    patients = sorted(pop.patient_id.unique())
    draws = rng.multinomial(N_PAT, [1 / N_PAT] * N_PAT, size=N_REPS).astype(np.int64)
    buf = io.BytesIO()
    np.save(buf, draws)
    (ev / 'patient_draws.npy').write_bytes(buf.getvalue())
    summ = {'patient_draws_patients': patients,
            'population': {'sha256': sha((popd / 'retrieval_population.csv').read_bytes())},
            'outputs': {'ranked_lists.csv.gz': {'sha256': sha((ev / 'ranked_lists.csv.gz').read_bytes())},
                        'query_metrics.csv.gz': {'sha256': sha((ev / 'query_metrics.csv.gz').read_bytes())},
                        'patient_draws.npy': {'sha256': sha(buf.getvalue())}}}
    (ev / 'summary.json').write_text(json.dumps(summ))
    for b, bd in A5.BUDGET_DIR.items():
        for F in range(5):
            d = root / 'features_qf' / bd / f'fold{F}'
            d.mkdir(parents=True)
            rows = []
            for i, s in enumerate(pop.slide_id):
                own = int(i % 5 == F)
                written = 0 if (s in NONRET[b] and own) else 1
                if s in NONRET[b] and not own and s == 1020:
                    written = 0  # not written in another fold's library: not a non-returned query
                rows.append({'slide_index': i, 'slide_id': s, 'patient_id': pop.patient_id[i],
                             'diagnostic_group': pop.diagnostic_group[i], 'slide_fold': i % 5, 'head_fold': F,
                             'budget_percent': int(float(b)), 'original_tiles': 100, 'retained_tiles': 50,
                             'written': written, 'reason': '' if written else 'retained_tiles<8', 'output_bytes': 1,
                             'sha256': 'x', 'own_fold_query': own})
            pd.DataFrame(rows).to_csv(d / '_filter_summary.csv', index=False)
    pins = {'summary': sha((ev / 'summary.json').read_bytes()),
            'ranked_lists': sha((ev / 'ranked_lists.csv.gz').read_bytes()),
            'query_metrics': sha((ev / 'query_metrics.csv.gz').read_bytes()),
            'expected_nonreturn': {b: sorted(v) for b, v in NONRET.items()}}
    cfg = A5.routeb_config(M, str(root), str(out_real), pins)
    cfg['pins']['population'] = sha((popd / 'retrieval_population.csv').read_bytes())
    cfg['pins']['patient_draws'] = sha(buf.getvalue())
    cfg['census_pins'] = None
    cfg['repo'] = REPO
    return {'root': root, 'out_real': out_real, 'cfg': cfg, 'pins': pins, 'pop': pop}


SIZES = (N_SLIDES, N_PAT, len(GROUPS), N_REPS)


def _guard_enc_s5(M, monkeypatch):
    class Boom:
        def __contains__(self, item):
            raise AssertionError('ENC_S5-keyed path called for route-(b) rows')

        def __getitem__(self, item):
            raise AssertionError('ENC_S5-keyed path called for route-(b) rows')

    monkeypatch.setattr(M, 'ENC_S5', Boom())
    for name in ('run', 'contrast_specs', 'method_contrasts', 'reconcile', 'lineage_check',
                 'submitted_error_reduction', 'input_paths'):
        monkeypatch.setattr(M, name, lambda *a, _n=name, **k: (_ for _ in ()).throw(
            AssertionError(f'frozen route-(a) {_n} called')))


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------

def test_frozen_loader_verifies_sha_and_stays_private(tmp_path):
    before = sha(open(FROZEN, 'rb').read())
    mod, info = A5.load_frozen_rer()
    assert info['sha256'] == before == A5.FROZEN_RER_SHA256
    assert 'r2_1_filter_rer' not in sys.modules
    bad = tmp_path / 'r2_1_filter_rer.py'
    bad.write_bytes(open(FROZEN, 'rb').read() + b'\n')
    with pytest.raises(RuntimeError, match='sha256'):
        A5.load_frozen_rer(str(bad))
    assert sha(open(FROZEN, 'rb').read()) == before


def test_config_constants_imported_not_retyped(M):
    assert M.K == 5 and M.METHODS == ('ot_cosine', 'yottixel') and M.UNDEFINED_THRESHOLD == 0.005
    assert M.max_undefined_reps(M.N_REPS) == 50 and M.QUANTILE_METHOD == 'linear'
    src = open(SCRIPT).read()
    for literal in ('UNDEFINED_THRESHOLD =', 'QUANTILE_METHOD =', 'K = 5', "METHODS = ("):
        assert literal not in src


def test_columns_equal_run001_header():
    p = os.path.join(OUT_REAL, 'r2_1_filter_rer_20261012', 'run_001', 'contrasts.csv')
    if not os.path.exists(p):
        pytest.skip('run_001 not present')
    with open(p) as f:
        header = f.readline().rstrip('\n').split(',')
    assert header == A5.CONTRAST_COLUMNS and len(header) == 27


def test_routeb_synthetic_schema_nan_submitted_and_no_enc_s5(world, M, monkeypatch, tmp_path):
    _guard_enc_s5(M, monkeypatch)
    out = world['root'] / 'rer_run_001'
    out.mkdir()
    s = A5.run_pipeline(M, world['cfg'], str(out), bootstrap=True, sizes=SIZES)
    con = pd.read_csv(io.BytesIO(s['outputs']['contrasts.csv']), dtype={'budget': str})
    assert list(con.columns) == A5.CONTRAST_COLUMNS
    assert len(con) == 8 * 2 * 2 * 15 == 480
    assert con['submitted_delta'].isna().all()                          # M2: all route-(b) submitted_delta NaN
    raw = pd.read_csv(io.BytesIO(s['outputs']['contrasts.csv']), dtype=str, keep_default_na=False)
    assert (raw['submitted_delta'] == '').all() and (raw['submitted_population_note'] == '').all()
    assert 'source_run' not in con.columns
    assert set(con.encoder) == {'uni2'} and set(con.design) == {'query_only', 'both_filtered'}
    assert set(con.budget) == {'5.0', '100.0'}
    assert set(con.contrast_id) == {f'uni2:{m}:{d}:{b}-vs-all_tissue' for m in ('ot_cosine', 'yottixel')
                                    for d in ('query_only', 'both_filtered') for b in ('5.0', '100.0')}
    assert not con.contrast_id.str.contains('legacy').any()
    for name in ('reconciliation.json', 'lineage_check.json'):
        assert json.loads(s['outputs'][name]) == {'not_applicable': 'route (b) has no submitted counterpart'}
    mc = pd.read_csv(io.BytesIO(s['outputs']['method_contrasts.csv']))
    assert len(mc) == 4 * 2 * 2 and set(mc.interval_label) == {A5.ROUTEB_METHOD_INTERVAL_LABEL}
    nf = con[(con.handling == 'miss') & (con['class'] == 'MACRO')]
    assert set(zip(nf.budget, nf.n_nonreturned_filt)) == {('5.0', 2), ('100.0', 3)}
    assert set(s['outputs']) >= {'estimates.csv', 'contrasts.csv', 'method_contrasts.csv', 'tie_sensitivity.csv',
                                 'tie_counts.csv', 'per_query_predictions.csv.gz', 'input_bindings.json',
                                 'README.md'}
    A5._finish(M, str(out), s, world['cfg'], {'bootstrap': True, 'frozen': {'sha256': A5.FROZEN_RER_SHA256}})
    comp = json.loads((out / 'COMPLETE.json').read_text())
    assert comp['source_run'] == 'route_b_uni2' and comp['encoder'] == 'uni2'
    assert 'route (b)' in (out / 'README.md').read_text()
    sums = (out / 'SHA256SUMS.txt').read_text().splitlines()
    for line in sums:
        h, n = line.split('  ')
        assert sha((out / n).read_bytes()) == h


def test_bootstrap_arithmetic_matches_frozen_kernels(world, M):
    """The contrast loop must reuse frozen kernels exactly: recompute one contrast independently."""
    out = world['root'] / 'rer_run_001'
    out.mkdir()
    s = A5.run_pipeline(M, world['cfg'], str(out), bootstrap=True, sizes=SIZES)
    con = pd.read_csv(io.BytesIO(s['outputs']['contrasts.csv']), dtype={'budget': str}, float_precision='round_trip')
    row = con[(con.method == 'ot_cosine') & (con.design == 'both_filtered') & (con.budget == '100.0')
              & (con.rule == 'plurality') & (con.handling == 'miss') & (con['class'] == 'MACRO')].iloc[0]
    pq = pd.read_csv(io.BytesIO(gzip.decompress(s['outputs']['per_query_predictions.csv.gz'])),
                     dtype={'budget': str}, keep_default_na=False)
    sel = pq[(pq.contrast_id == row.contrast_id) & (pq.rule == 'plurality') & (pq.handling == 'miss')]
    pop = world['pop']
    groups = sorted(GROUPS)
    order = {s_: i for i, s_ in enumerate(pop.slide_id)}
    T = np.zeros((len(pop), len(groups)))
    T[np.arange(len(pop)), [groups.index(g) for g in pop.diagnostic_group]] = 1
    PA = M.prediction_matrix(list(sel[sel.arm == 'all_tissue'].sort_values('query', key=lambda x: x.map(order))
                                  .prediction), groups)
    PF = M.prediction_matrix(list(sel[sel.arm == 'both_filtered'].sort_values('query', key=lambda x: x.map(order))
                                  .prediction), groups)
    fA, _ = M.weighted_f1(np.ones((1, len(pop))), T, PA, np.ones(len(pop)))
    fF, _ = M.weighted_f1(np.ones((1, len(pop))), T, PF, np.ones(len(pop)))
    assert row.f1_all == float(M.macro_mean(fA[0])[0])
    assert row.f1_filt == float(M.macro_mean(fF[0])[0])


def test_nonreturn_pin_must_equal_census(world, M):
    cfg = dict(world['cfg'])
    enr = {b: set(v) for b, v in NONRET.items()}
    enr['5.0'] = enr['5.0'] | {1020}
    cfg['expected_nonreturn'] = A5.expected_nonreturn_table('uni2', enr)
    out = world['root'] / 'rer_run_001'
    out.mkdir()
    with pytest.raises(M.GateError) as ex:
        A5.run_pipeline(M, cfg, str(out), bootstrap=False, sizes=SIZES)
    assert ex.value.gate == 'nonreturn_census'


def test_evaluator_nonreturn_must_equal_pin(world, M):
    ev = world['root'] / 'evaluation_qf'
    qm = pd.read_csv(ev / 'query_metrics.csv.gz', dtype={'budget': str})
    rl = pd.read_csv(ev / 'ranked_lists.csv.gz', dtype={'budget': str})
    # make query 1011 returned at budget 5 in the evaluator (census still says non-returned)
    src = rl[(rl['query'] == 1012) & (rl.arm == 'query_only') & (rl.budget == '5')].copy()
    m = (qm['query'] == 1011) & (qm.arm == 'query_only') & (qm.budget == '5')
    qm.loc[m, 'status'] = 'returned'
    for meth in ('ot_cosine', 'yottixel'):
        add = src[src.method == meth].copy()
        add['query'] = 1011
        pat = world['pop'].set_index('slide_id').patient_id
        if (add.neighbour.map(pat) == pat[1011]).any():
            pytest.skip('synthetic neighbour collision')
        grp = world['pop'].set_index('slide_id').diagnostic_group[1011]
        add['relevant'] = (add.neighbour_group == grp).astype(int)
        rl = pd.concat([rl, add])
        hit = float(any(add.sort_values('rank').neighbour_group.iloc[:5] == grp))
        mm = m & (qm.method == meth) & (qm.k == 5)
        qm.loc[mm, 'conditional_hit'] = hit
        qm.loc[mm, 'boundary_tie'] = int(add.sort_values('rank').distance.iloc[4] ==
                                         add.sort_values('rank').distance.iloc[5])
    (ev / 'query_metrics.csv.gz').write_bytes(_gz(qm))
    (ev / 'ranked_lists.csv.gz').write_bytes(_gz(rl))
    cfg = dict(world['cfg'])
    cfg['pins'] = dict(cfg['pins'], query_metrics=sha((ev / 'query_metrics.csv.gz').read_bytes()),
                       ranked_lists=sha((ev / 'ranked_lists.csv.gz').read_bytes()))
    out = world['root'] / 'rer_run_001'
    out.mkdir()
    with pytest.raises(M.GateError) as ex:
        A5.run_pipeline(M, cfg, str(out), bootstrap=False, sizes=SIZES)
    assert ex.value.gate == 'nonreturn_sets'


def test_input_hash_mismatch_is_a_gate(world, M):
    cfg = dict(world['cfg'])
    cfg['pins'] = dict(cfg['pins'], patient_draws=A5.DRAWS_SHA256)  # the production draws pin; synthetic differs
    out = world['root'] / 'rer_run_001'
    out.mkdir()
    with pytest.raises(M.GateError) as ex:
        A5.run_pipeline(M, cfg, str(out), bootstrap=False, sizes=SIZES)
    assert ex.value.gate == 'input_hashes'


def test_production_config_pins_draws_and_population(M, tmp_path):
    pins = {'summary': 'a', 'ranked_lists': 'b', 'query_metrics': 'c', 'expected_nonreturn': {'5.0': [1], '100.0': []}}
    cfg = A5.routeb_config(M, str(tmp_path / 'r'), str(tmp_path), pins)
    assert cfg['pins']['patient_draws'] == 'cfaed70d0c11084fb72b4dd7c787313b5da9d6e47f374803d2671b647886f350'
    assert cfg['pins']['population'] == 'c7cae1f0d855952e84e85aa90c7ef7d8a622086273c113285655c96be946d97b'
    assert cfg['pins']['vote_module'] == M.PIN['vote_module']
    assert cfg['paths']['patient_draws'].endswith(os.path.join('r', 'evaluation_qf', 'patient_draws.npy'))
    assert cfg['expected_nonreturn'][('uni2', 'all_tissue', '')] == set()
    assert cfg['expected_nonreturn'][('uni2', 'both_filtered', '5.0')] == {1}
    assert cfg['source_run'] == 'route_b_uni2'


def test_derive_pins_and_load_pins(world, tmp_path):
    root = str(world['root'])
    # derive_pins checks the production draws/population pins: the synthetic summary must fail closed ...
    with pytest.raises(ValueError, match='patient_draws'):
        A5.derive_pins(root)
    # ... and the census part alone reproduces the synthetic non-return sets
    census, bind = A5.nonreturn_from_census(os.path.join(root, 'features_qf'))
    assert census == NONRET and len(bind) == 10
    p = tmp_path / 'pins.json'
    p.write_text(json.dumps(world['pins']))
    pins, info = A5.load_pins(str(p), sha(p.read_bytes()))
    assert info['sha256'] == sha(p.read_bytes())
    with pytest.raises(ValueError, match='sha256'):
        A5.load_pins(str(p), '0' * 64)


def test_census_reproduces_frozen_hoptimus_expected_nonreturn(M):
    root = os.path.join(OUT_REAL, 'r2_1_filter_20261003', 'features_qf')
    if not os.path.isdir(root):
        pytest.skip('H-Optimus-1 census not present')
    census, _ = A5.nonreturn_from_census(root)
    for b in A5.BUDGETS:
        for d in A5.DESIGNS:
            assert census[b] == M.EXPECTED_NONRETURN[('hoptimus', d, b)]


def test_r0_compare_selection_and_negative_control(world, M, tmp_path):
    out = world['root'] / 'rer_run_001'
    out.mkdir()
    cfg = dict(world['cfg'])
    s = A5.run_pipeline(M, cfg, str(out), bootstrap=True, sizes=SIZES)
    # synthetic "run_001": other-encoder rows first, then these rows, written by the same writer
    run001 = tmp_path / 'run_001'
    run001.mkdir()
    pins = {}
    for name in ('contrasts.csv', 'estimates.csv'):
        df = pd.read_csv(io.BytesIO(s['outputs'][name]), dtype=str, keep_default_na=False)
        other = df.copy()
        other['encoder'] = 'virchow2'
        b = M._csv_bytes(pd.concat([other, df]))
        (run001 / name).write_bytes(b)
        pins[name] = sha(b)
    res = A5.r0_compare(M, str(run001), s['outputs'], encoder='uni2', pins=pins, diagnostic_pins={})
    assert res['pass'], res
    bad = dict(s['outputs'])
    con = pd.read_csv(io.BytesIO(bad['contrasts.csv']), dtype=str, keep_default_na=False)
    con.loc[0, 'f1_all'] = con.loc[0, 'f1_all'] + '1'
    bad['contrasts.csv'] = M._csv_bytes(con)
    res = A5.r0_compare(M, str(run001), bad, encoder='uni2', pins=pins, diagnostic_pins={})
    assert not res['pass'] and not res['gate_files']['contrasts.csv']['byte_identical']
    res = A5.r0_compare(M, str(run001), s['outputs'], encoder='uni2',
                        pins={k: '0' * 64 for k in pins}, diagnostic_pins={})
    assert not res['pass']


def test_r0_config_uses_frozen_hoptimus_pins(M, tmp_path):
    cfg = A5.r0_config(M, str(tmp_path))
    assert cfg['encoder'] == 'hoptimus' and cfg['source_run'] == 'r0_hoptimus_regression'
    assert cfg['pins']['ranked_lists'] == M.PIN['ranked_lists:hoptimus']
    assert cfg['pins']['query_metrics'] == M.PIN['query_metrics:hoptimus']
    assert cfg['expected_nonreturn'][('hoptimus', 'query_only', '100.0')] == {14601, 15678, 15794, 16737, 16738}
    assert cfg['method_interval_label'] == 'descriptive (item 8c, H-Optimus)'
    assert [s['contrast_id'] for s in A5.contrast_specs('hoptimus', M)] == \
        [s['contrast_id'] for s in M.contrast_specs() if s['encoder'] == 'hoptimus']


def test_cli_usage_and_failed_paths(tmp_path, monkeypatch, capsys):
    out_root = tmp_path / 'OUT'
    root = out_root / A5.ROOT_NAME
    root.mkdir(parents=True)
    monkeypatch.delenv('PROSPER_OUT', raising=False)
    assert A5.main(['run', '--out', str(root / 'rer_run_001'), '--pins', 'x', '--pins-sha256', 'y']) == 2
    monkeypatch.setenv('PROSPER_OUT', str(out_root))
    assert A5.main(['run', '--out', str(tmp_path / 'elsewhere'), '--pins', 'x', '--pins-sha256', 'y']) == 2
    assert A5.main(['r0', '--out', str(root / 'rer_x')]) == 2              # R-0 must be under ROOT/fidelity/r0
    pins = tmp_path / 'pins.json'
    pins.write_text('{}')
    rc = A5.main(['run', '--out', str(root / 'rer_run_001'), '--pins', str(pins), '--pins-sha256', '0' * 64])
    assert rc == 1
    failed = json.loads((root / 'rer_run_001' / 'FAILED.json').read_text())
    assert failed['status'] == 'FAILED' and 'sha256' in failed['reason']
    assert not (root / 'rer_run_001' / 'COMPLETE.json').exists()
    assert A5.main(['run', '--out', str(root / 'rer_run_001'), '--pins', str(pins), '--pins-sha256', 'y']) == 2


# ----------------------------------------------------------------------------------------------
# AMENDMENT_002: tie-block containment evidence from the frozen no-same-patient matrix row
# ----------------------------------------------------------------------------------------------

TQ, TM, TA, TB = 1001, 'yottixel', 'query_only', '100'   # synthetic triggering list (no k%7 tie in the base world)
TD = [0.1, 0.2, 0.3, 0.4] + [0.5] * 6                    # ranks 5-10 tied: block reaches the stored depth 10
T_NAN = (1060, 1062)                                     # non-library (NaN) cells of the TQ row


def _neighbours(pop):
    pat = dict(zip(pop.slide_id, pop.patient_id))
    return [int(s) for s in pop.slide_id if pat[s] != pat[TQ]][:10]


def _tie_world(world, M, mutate=None, summary_sha=None, ab_sha=None):
    """Make (yottixel, query_only, 100, query 1001) a depth-reaching block and write its synthetic qf matrix."""
    root, pop = world['root'], world['pop']
    ev = root / 'evaluation_qf'
    group_of = dict(zip(pop.slide_id, pop.diagnostic_group))
    pat_of = dict(zip(pop.slide_id, pop.patient_id))
    nb = _neighbours(pop)
    rl = pd.read_csv(ev / 'ranked_lists.csv.gz', dtype={'budget': str})
    qm = pd.read_csv(ev / 'query_metrics.csv.gz', dtype={'budget': str})
    m = (rl.method == TM) & (rl.arm == TA) & (rl.budget == TB) & (rl['query'] == TQ)
    assert list(rl.loc[m, 'rank']) == list(range(1, 11))
    rl.loc[m, 'neighbour'] = nb
    rl.loc[m, 'neighbour_group'] = [group_of[x] for x in nb]
    rl.loc[m, 'relevant'] = [int(group_of[x] == group_of[TQ]) for x in nb]
    rl.loc[m, 'distance'] = TD
    mq = (qm.method == TM) & (qm.arm == TA) & (qm.budget == TB) & (qm['query'] == TQ) & (qm.k == 5)
    qm.loc[mq, 'boundary_tie'] = 1
    qm.loc[mq, 'conditional_hit'] = float(any(group_of[x] == group_of[TQ] for x in nb[:5]))
    (ev / 'ranked_lists.csv.gz').write_bytes(_gz(rl))
    (ev / 'query_metrics.csv.gz').write_bytes(_gz(qm))

    ids = [int(s) for s in pop.slide_id]
    rng = np.random.default_rng(11)
    vals = rng.uniform(0.6, 0.99, size=(len(ids), len(ids)))
    for r, q in enumerate(ids):
        for c, s in enumerate(ids):
            if pat_of[s] == pat_of[q]:
                vals[r, c] = A5.SENTINEL                  # self and same-patient cells
    col = {s: j for j, s in enumerate(ids)}
    r = col[TQ]
    for s in T_NAN:
        vals[r, col[s]] = np.nan
    for x, dd in zip(nb, TD):
        vals[r, col[x]] = dd
    if mutate == 'extra_eq':
        vals[r, col[1030]] = 0.5                          # an 11th library entry at d5: block extends beyond rank 10
    elif mutate == 'value_mismatch':
        vals[r, col[nb[6]]] = 0.55                        # matrix row disagrees with the stored list
    elif mutate == 'smaller_unstored':
        vals[r, col[1030]] = 0.05                         # a library entry below the stored rank-1 distance
    df = pd.DataFrame(vals, index=[str(s) for s in ids], columns=[str(s) for s in ids])
    qf = root / 'similarity_matrix_qf'
    qf.mkdir(exist_ok=True)
    path = qf / A5.QF_MATRIX_NAME.format(b=TB, d=A5.DESIGN_TOKEN[TA], m=TM)
    df.to_pickle(path)
    msha = sha(path.read_bytes())
    summ = json.loads((ev / 'summary.json').read_text())
    summ['outputs']['ranked_lists.csv.gz']['sha256'] = sha((ev / 'ranked_lists.csv.gz').read_bytes())
    summ['outputs']['query_metrics.csv.gz']['sha256'] = sha((ev / 'query_metrics.csv.gz').read_bytes())
    summ['arms'] = [{'key': f'{TA}|{TM}|{TB}', 'matrix': str(path), 'matrix_sha256': summary_sha or msha}]
    (ev / 'summary.json').write_text(json.dumps(summ))
    (ev / 'arm_bindings.json').write_text(json.dumps({'arms': [
        {'arm': TA, 'method': TM, 'budget': TB, 'matrix': str(path), 'matrix_sha256': ab_sha or msha}]}))
    cfg = dict(world['cfg'])
    cfg['pins'] = dict(cfg['pins'], summary=sha((ev / 'summary.json').read_bytes()),
                       ranked_lists=sha((ev / 'ranked_lists.csv.gz').read_bytes()),
                       query_metrics=sha((ev / 'query_metrics.csv.gz').read_bytes()))
    return cfg, nb, msha, str(path)


def _frozen_reason(M):
    with pytest.raises(M.GateError) as ex:
        M.reverse_boundary_block(['a'] * 10, np.array(TD))
    assert ex.value.gate == 'tie_block_containment' and ex.value.detail is None
    return ex.value.reason


def test_a002_contained_block_at_stored_depth_is_reversed(world, M, monkeypatch):
    """(i) block ends exactly at the stored depth -> contained, frozen reversal applied; (iv) at pipeline level."""
    cfg, nb, msha, path = _tie_world(world, M)
    group_of = dict(zip(world['pop'].slide_id, world['pop'].diagnostic_group))
    calls = []
    frozen_vote = M.vote
    monkeypatch.setattr(M, 'vote', lambda labs, *a: (calls.append(tuple(labs)), frozen_vote(labs, *a))[1])
    out = world['root'] / 'rer_run_002'
    out.mkdir()
    s = A5.run_pipeline(M, cfg, str(out), bootstrap=True, sizes=SIZES)
    assert M.reverse_boundary_block is M._routeb_frozen_reverse_boundary_block      # uninstalled afterwards
    ev = json.loads(s['outputs'][A5.RESOLUTION_FILE])
    assert ev['amendment'] == '002' and ev['matrix_resolution_enabled'] is True and ev['n_activated'] == 1
    (rec,) = ev['activated']
    assert (rec['arm'], rec['method'], rec['budget'], rec['query']) == (TA, TM, '100.0', TQ)
    assert (rec['lo'], rec['hi'], rec['n_lt'], rec['n_eq'], rec['d5']) == (4, 9, 4, 6, 0.5)
    assert rec['n_library'] == N_SLIDES - 2 - len(T_NAN)   # self + same-patient 1e6 and NaN cells ignored
    assert rec['decision'] == 'contained' and sorted(rec['rules']) == sorted(M.RULES)
    assert rec['matrix'] == {'path': path, 'sha256': msha} and rec['stored_depth'] == 10
    tot = {(t['method'], t['arm'], t['budget']): t for t in ev['totals']}
    assert len(tot) == 10
    assert tot[(TM, TA, '100.0')]['block_reaches_stored_depth'] == 1 == tot[(TM, TA, '100.0')]['resolved_by_matrix']
    assert sum(t['block_reaches_stored_depth'] for t in ev['totals']) == 1
    ib = json.loads(s['outputs']['input_bindings.json'])
    (bk,) = ib['tie_block_matrices']
    assert bk == f'{TA}|{TM}|{TB}' and ib['tie_block_matrices'][bk]['sha256'] == msha
    assert ib['tie_block_matrices'][bk]['match'] is True and 'arm_bindings_sha256' in ib['tie_block_matrices'][bk]
    assert 'tie_block_matrix_evidence' in s['gates'] and 'amendment 002' in s['gates']['tie_block_containment']
    plain = tuple(group_of[x] for x in nb[:5])
    rev = tuple(group_of[x] for x in nb[:4]) + (group_of[nb[9]],)
    assert plain != rev and plain in calls and rev in calls       # frozen block reversal on the stored labels
    A5._finish(M, str(out), s, cfg, {'bootstrap': True, 'frozen': {'sha256': A5.FROZEN_RER_SHA256}})
    comp = json.loads((out / 'COMPLETE.json').read_text())
    assert comp['amendment'] == '002' and A5.RESOLUTION_FILE in comp['outputs']
    assert json.loads((out / A5.RESOLUTION_FILE).read_text()) == ev


def test_a002_block_beyond_stored_depth_fails_with_same_gate(world, M):
    """(ii) an 11th library entry at d5 -> tie_block_containment with the frozen reason (FAILED)."""
    reason = _frozen_reason(M)
    cfg, _, _, _ = _tie_world(world, M, mutate='extra_eq')
    out = world['root'] / 'rer_run_002'
    out.mkdir()
    with pytest.raises(M.GateError) as ex:
        A5.run_pipeline(M, cfg, str(out), bootstrap=False, sizes=SIZES)
    assert ex.value.gate == 'tie_block_containment' and ex.value.reason == reason
    assert ex.value.detail['decision'] == 'not_contained' and (ex.value.detail['n_lt'], ex.value.detail['n_eq']) == \
        (4, 7)
    assert M.reverse_boundary_block is M._routeb_frozen_reverse_boundary_block
    A5._write_failed(M, str(out), ex.value.gate, ex.value.reason, ex.value.detail, A5.run_pipeline.state)
    failed = json.loads((out / 'FAILED.json').read_text())
    assert failed['gate'] == 'tie_block_containment' and failed['input_bindings']['tie_block_matrices']
    assert not (out / 'COMPLETE.json').exists()


@pytest.mark.parametrize('mutate,kw,gate', [
    ('value_mismatch', {}, 'tie_block_matrix_consistency'),
    ('smaller_unstored', {}, 'tie_block_matrix_consistency'),
    (None, {'summary_sha': '0' * 64}, 'tie_block_matrix_binding'),
    (None, {'ab_sha': '1' * 64}, 'tie_block_matrix_binding'),
])
def test_a002_matrix_disagreement_fails(world, M, mutate, kw, gate):
    """(iii) matrix row / stored list disagreement or matrix custody mismatch -> FAILED, never resolved."""
    cfg, _, _, _ = _tie_world(world, M, mutate=mutate, **kw)
    out = world['root'] / 'rer_run_002'
    out.mkdir()
    with pytest.raises(M.GateError) as ex:
        A5.run_pipeline(M, cfg, str(out), bootstrap=False, sizes=SIZES)
    assert ex.value.gate == gate
    assert M.reverse_boundary_block is M._routeb_frozen_reverse_boundary_block


def test_a002_counts_ignore_sentinel_nan_and_inf():
    """(iv) only finite non-sentinel library entries count."""
    row = [0.1, 0.2, np.nan, 1e6, 0.5, 0.5, np.inf, -np.inf, 0.3, np.nan, 1e6]
    assert A5.matrix_block_counts(row, 0.5) == (3, 2, 5)
    assert A5.matrix_block_counts([1e6, 1e6, 0.2, np.nan], 1e6) == (1, 0, 1)   # sentinel never counts as == d5
    assert A5.matrix_block_counts(np.array([np.nan] * 3), 0.5) == (0, 0, 0)


@pytest.mark.parametrize('d,expect', [
    (TD, (4, 9)), ([.1, .2, .3, .5, .5, .5, .5, .5, .5, .5], (3, 9)), ([.1] * 10, (0, 9)),
    ([.1, .2, .3, .4, .5, .5, .6, .7, .8, .9], None), ([.1, .2, .3, .4, .5, .6, .7, .8, .9, .9], None),
    ([.1, .2, .3, .4, .5, .5], (4, 5)), ([.1, .2, .3, .4, .5], None)])
def test_a002_reaches_depth_mirrors_frozen_branch(M, d, expect):
    assert A5.block_reaches_stored_depth(d, M.K) == expect
    if expect is None:
        M.reverse_boundary_block(['x'] * len(d), np.array(d))
    else:
        with pytest.raises(M.GateError, match='stored depth'):
            M.reverse_boundary_block(['x'] * len(d), np.array(d))


def test_a002_r0_configuration_never_activates(world, M):
    """(v) r0 config disables the path: the frozen error is raised unchanged and no matrix is read."""
    assert A5.r0_config(M, '/nonexistent')['tie_block_matrix_resolution'] is False
    assert A5.routeb_config(M, '/r', '/o', {'summary': 'a', 'ranked_lists': 'b', 'query_metrics': 'c',
                                           'expected_nonreturn': {'5.0': [], '100.0': []}}
                            )['tie_block_matrix_resolution'] is True
    reason = _frozen_reason(M)
    cfg, _, _, _ = _tie_world(world, M)
    cfg['tie_block_matrix_resolution'] = False
    out = world['root'] / 'rer_run_002'
    out.mkdir()
    with pytest.raises(M.GateError) as ex:
        A5.run_pipeline(M, cfg, str(out), bootstrap=False, sizes=SIZES)
    assert (ex.value.gate, ex.value.reason, ex.value.detail) == ('tie_block_containment', reason, None)
    assert A5.run_pipeline.state['tie_block_matrices'] == {}


def test_a002_context_required_for_activation(world, M):
    """Outside the driver's per-arm context the wrapper re-raises the frozen error unchanged."""
    import types
    pop = world['pop']
    r = A5.TieBlockResolver(M, True, pop, pd.DataFrame(), {}, {}).install()
    try:
        with pytest.raises(M.GateError) as ex:
            M.reverse_boundary_block(['a'] * 10, np.array(TD))
        assert ex.value.detail is None
        arm = {'lists': {1: (['a'] * 10, np.array(TD))}, 'nonret': np.zeros(len(pop), bool)}
        other = np.array(TD)                                         # equal values, not the stored list object
        with r.context(('uni2', TM, TA, '100.0'), arm, 'plurality'):
            with pytest.raises(M.GateError) as ex:
                M.reverse_boundary_block(['a'] * 10, other)
        assert ex.value.detail is None and r.activated == {}
    finally:
        r.uninstall()
    assert isinstance(M.reverse_boundary_block, types.FunctionType)


def test_a002_no_activation_outputs_and_byte_identity(world, M):
    """(vi) resolution JSON written with an empty list; outputs other than the new evidence are identical with the
    path enabled or disabled (R-0 compares contrasts/estimates and diagnostics only)."""
    outs = {}
    for flag in (True, False):
        cfg = dict(world['cfg'], tie_block_matrix_resolution=flag)
        out = world['root'] / f'rer_{int(flag)}'
        out.mkdir()
        outs[flag] = A5.run_pipeline(M, cfg, str(out), bootstrap=True, sizes=SIZES)
    s = outs[True]
    ev = json.loads(s['outputs'][A5.RESOLUTION_FILE])
    assert ev['activated'] == [] and ev['n_activated'] == 0 and len(ev['totals']) == 10
    assert sum(t['boundary_tie_queries'] for t in ev['totals']) > 0
    assert sum(t['block_reaches_stored_depth'] for t in ev['totals']) == 0
    assert json.loads(s['outputs']['input_bindings.json'])['tie_block_matrices'] == {}
    assert s['gates']['tie_block_containment'] == 'every k=5 boundary tie block lies within the stored top-10'
    assert 'tie_block_matrix_evidence' not in s['gates']
    for name in set(s['outputs']) - {A5.RESOLUTION_FILE, 'input_bindings.json'}:
        assert s['outputs'][name] == outs[False]['outputs'][name], name
    assert s['gates'] == outs[False]['gates']
    out = world['root'] / 'rer_1'
    A5._finish(M, str(out), s, world['cfg'], {'bootstrap': True, 'frozen': {'sha256': A5.FROZEN_RER_SHA256}})
    assert json.loads((out / 'COMPLETE.json').read_text())['amendment'] == '002'
    assert json.loads((out / A5.RESOLUTION_FILE).read_text())['activated'] == []


# ---- real data (read-only; skipped when absent) ----

REAL_ROOT = os.path.join(OUT_REAL, 'r2_1_route_b_uni2_20261013')


@pytest.mark.parametrize('arm', ['both_filtered', 'query_only'])
def test_a002_real_triggering_row_4379_is_contained(arm):
    """The actual rer_run_001 failure: budget100 yottixel query 4379, ranks 5-10 tied at 354 (the only two
    route-(b) lists whose block reaches the stored depth)."""
    ev = os.path.join(REAL_ROOT, 'evaluation_qf')
    path = os.path.join(REAL_ROOT, 'similarity_matrix_qf',
                        A5.QF_MATRIX_NAME.format(b='100', d=A5.DESIGN_TOKEN[arm], m='yottixel'))
    if not (os.path.exists(path) and os.path.exists(os.path.join(ev, 'summary.json'))):
        pytest.skip('route-(b) qf matrix not present')
    with open(path, 'rb') as f:
        data = f.read()
    summ = json.load(open(os.path.join(ev, 'summary.json')))
    (entry,) = [a for a in summ['arms'] if a['key'] == f'{arm}|yottixel|100']
    assert sha(data) == entry['matrix_sha256'] and os.path.samefile(entry['matrix'], path)
    df = pd.read_pickle(io.BytesIO(data))
    row = df.loc['4379'].to_numpy(dtype=np.float64)
    assert A5.matrix_block_counts(row, 354.0) == (4, 6, A5.matrix_block_counts(row, 354.0)[2])
    rl = pd.read_csv(os.path.join(ev, 'ranked_lists.csv.gz'), dtype={'budget': str})
    g = rl[(rl.method == 'yottixel') & (rl.arm == arm) & (rl.budget == '100') & (rl['query'] == 4379)]
    g = g.sort_values('rank')
    d = g.distance.to_numpy(dtype=np.float64)
    assert d[4] == 354.0 and A5.block_reaches_stored_depth(d, 5) == (4, 9)
    lib = np.isfinite(row) & (row != A5.SENTINEL)
    assert A5._same_bits(np.sort(row[lib], kind='mergesort')[:10], d)
    assert set(str(x) for x in g.neighbour) <= set(df.columns[lib])
    reach = [k for k, gg in rl.groupby(['method', 'arm', 'budget', 'query'], dropna=False, sort=False)
             if A5.block_reaches_stored_depth(gg.sort_values('rank').distance.to_numpy(dtype=np.float64), 5)]
    assert sorted(reach) == [('yottixel', 'both_filtered', '100', 4379), ('yottixel', 'query_only', '100', 4379)]


def test_a002_real_hoptimus_r0_lists_never_reach_stored_depth(M):
    """(v) on the actual R-0 input: no H-Optimus-1 boundary block reaches the stored depth."""
    cfg = A5.r0_config(M, OUT_REAL)
    p = cfg['paths']['ranked_lists']
    if not os.path.exists(p):
        pytest.skip('H-Optimus-1 evaluation_qf not present')
    with open(p, 'rb') as f:
        assert sha(f.read()) == cfg['pins']['ranked_lists']
    rl = pd.read_csv(p, dtype={'budget': str})
    n_tie = n_reach = 0
    for _, g in rl.groupby(['method', 'arm', 'budget', 'query'], dropna=False, sort=False):
        d = g.sort_values('rank').distance.to_numpy(dtype=np.float64)
        n_tie += M.has_boundary_tie(d)
        n_reach += A5.block_reaches_stored_depth(d, M.K) is not None
    assert n_tie > 0 and n_reach == 0
