"""Synthetic-fixture tests for scripts/r2_1_filter_rer.py (R2.1 B(c) fair-design RER).

No real data are read. The frozen vote module (code, not data) is loaded through the script's own
hash-verified loader. Run with an explicit --basetemp outside /tmp and without the cache provider.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, 'scripts', 'r2_1_filter_rer.py')
VOTE_MODULE = os.path.join(REPO, 'scripts', 'r2_3_internal_failure_tables.py')


def _load_script():
    spec = importlib.util.spec_from_file_location('r2_1_filter_rer_under_test', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_script()


@pytest.fixture(scope='module')
def votes():
    plurality, legacy_generous, info = M.load_vote_functions(VOTE_MODULE)
    assert info['sha256'] == M.PIN['vote_module']
    return plurality, legacy_generous


def onehot(labels, groups):
    gi = {g: i for i, g in enumerate(groups)}
    T = np.zeros((len(labels), len(groups)))
    for i, l in enumerate(labels):
        T[i, gi[l]] = 1.0
    return T


# ----------------------------------------------------------------------------------------------
# (a) F1 / macro / RER arithmetic
# ----------------------------------------------------------------------------------------------

def test_f1_convention_and_zero_tp():
    f1 = M.f1_from_counts([3, 0, 0, 0], [1, 2, 0, 0], [2, 4, 3, 0], [5, 4, 3, 0])
    assert f1[0] == pytest.approx(6 / 9)              # 2TP/(2TP+FP+FN)
    assert f1[1] == 0.0                               # zero TP with positive denominator -> 0
    assert f1[2] == 0.0                               # zero TP, support > 0, never predicted -> 0
    assert np.isnan(f1[3])                            # zero support -> undefined (excluded)


def test_f1_from_confusion_matches_counts():
    cm = np.array([[3, 1, 1], [0, 2, 0], [1, 0, 0]])
    f1 = M.f1_from_confusion(cm)
    assert f1[0] == pytest.approx(2 * 3 / (2 * 3 + 1 + 2))
    assert f1[1] == pytest.approx(2 * 2 / (2 * 2 + 1 + 0))
    assert f1[2] == 0.0


def test_macro_and_rer_arithmetic():
    m, nex = M.macro_mean(np.array([[0.5, np.nan, 1.0], [0.2, 0.4, 0.6]]))
    assert m[0] == pytest.approx(0.75) and nex[0] == 1
    assert m[1] == pytest.approx(0.4) and nex[1] == 0
    r, u = M.rer_values(0.8, 0.6)
    assert float(r) == pytest.approx(0.5) and not bool(u)
    r, u = M.rer_values(0.5, 0.6)
    assert float(r) == pytest.approx(-0.25) and not bool(u)
    assert M.submitted_error_reduction(0.6, 0.8) == pytest.approx(0.5)
    assert np.isnan(M.submitted_error_reduction(1.0, 0.9))
    assert np.isnan(M.submitted_error_reduction(1.0 - 5e-9, 0.9))   # inside the submitted 1e-8 eps


def test_weighted_f1_equals_replicated_rows():
    groups = ['A', 'B']
    T = onehot(['A', 'A', 'B', 'B'], groups)
    P = onehot(['A', 'B', 'B', 'A'], groups)
    W = np.array([[2, 1, 0, 3]], float)
    f1, sup = M.weighted_f1(W, T, P, np.ones(4))
    # expanded: A,A (pred A,A), A (pred B), B,B,B (pred A)
    cmw = np.array([[2, 1], [3, 0]])
    assert np.allclose(f1[0], M.f1_from_confusion(cmw))
    assert np.allclose(sup[0], [3, 3])


# ----------------------------------------------------------------------------------------------
# (b) plurality vs legacy generous on tie fixtures
# ----------------------------------------------------------------------------------------------

def test_plurality_vs_generous_ties(votes):
    plurality, generous = votes
    labs = ['A', 'B', 'B', 'A', 'C']                  # 2-2 tie, A closest-ranked
    assert M.vote(labs, 'A', 'plurality', plurality, generous) == 'A'
    assert M.vote(labs, 'A', 'legacy_generous_ORACLE', plurality, generous) == 'A'
    assert M.vote(labs, 'B', 'plurality', plurality, generous) == 'A'
    assert M.vote(labs, 'B', 'legacy_generous_ORACLE', plurality, generous) == 'B'   # oracle picks truth
    assert M.vote(labs, 'C', 'legacy_generous_ORACLE', plurality, generous) == 'A'   # truth not tied
    clear = ['C', 'B', 'B', 'B', 'A']
    for truth in 'ABC':
        assert M.vote(clear, truth, 'plurality', plurality, generous) == 'B'
        assert M.vote(clear, truth, 'legacy_generous_ORACLE', plurality, generous) == 'B'
    with pytest.raises(M.GateError):
        M.vote(['A', 'B'], 'A', 'plurality', plurality, generous)


def test_vote_loader_rejects_wrong_bytes(tmp_path):
    p = tmp_path / 'fake_vote.py'
    p.write_bytes(b'def plurality(l, k):\n    return l[0]\n\ndef legacy_generous(l, t, k):\n    return True, t\n')
    with pytest.raises(M.GateError) as ex:
        M.load_vote_functions(str(p))
    assert ex.value.gate == 'vote_module_hash'
    # exactly the supplied bytes are executed when they carry the expected hash
    src = p.read_bytes()
    pl, lg, info = M.load_vote_functions(str(p), expected_sha=hashlib.sha256(src).hexdigest(), source=src)
    assert pl(['X', 'Y'], 2) == 'X' and info['sha256'] == hashlib.sha256(src).hexdigest()


# ----------------------------------------------------------------------------------------------
# (c) non-return handling: miss vs both_returned
# ----------------------------------------------------------------------------------------------

def test_nonreturn_miss_and_both_returned():
    groups = ['A', 'B']
    truth = ['A', 'A', 'B', 'B']
    T = onehot(truth, groups)
    PA = M.prediction_matrix(['A', 'A', 'B', 'B'], groups)          # all-tissue: perfect
    PF = M.prediction_matrix(['A', 'NONE', 'B', 'B'], groups)       # query 1 non-returned in filtered arm
    assert PF[1].sum() == 0
    nonret_all = np.zeros(4, bool)
    nonret_filt = np.array([False, True, False, False])

    inc = M.inclusion_mask('miss', nonret_all, nonret_filt)
    assert inc.tolist() == [1, 1, 1, 1]
    res = M.evaluate_contrast(None, T, PA, PF, inc)
    # miss: FN for A only (no FP for B): F1_A = 2/(2+0+1), F1_B = 1
    assert res['fF_pt'][0] == pytest.approx(2 / 3)
    assert res['fF_pt'][1] == pytest.approx(1.0)
    assert res['sup_pt'].tolist() == [2, 2]

    inc = M.inclusion_mask('both_returned', nonret_all, nonret_filt)
    assert inc.tolist() == [1, 0, 1, 1]
    res = M.evaluate_contrast(None, T, PA, PF, inc)
    assert res['fF_pt'].tolist() == [1.0, 1.0] and res['fA_pt'].tolist() == [1.0, 1.0]
    assert res['sup_pt'].tolist() == [1, 2]                         # dropped from both arms

    # a query non-returned in the all-tissue arm is also dropped from both arms
    inc = M.inclusion_mask('both_returned', np.array([False, False, True, False]), nonret_filt)
    assert inc.tolist() == [1, 0, 0, 1]
    with pytest.raises(ValueError):
        M.inclusion_mask('other', nonret_all, nonret_filt)
    with pytest.raises(M.GateError):
        M.prediction_matrix(['Z'], groups)


# ----------------------------------------------------------------------------------------------
# (d) zero-support class exclusion within a replicate
# ----------------------------------------------------------------------------------------------

def test_zero_support_class_excluded_in_replicate():
    groups = ['A', 'B', 'C']
    T = onehot(['A', 'B', 'C', 'C'], groups)
    PA = onehot(['A', 'B', 'C', 'A'], groups)
    PF = onehot(['A', 'B', 'C', 'C'], groups)
    W = np.array([[1, 1, 1, 1],        # full support
                  [1, 0, 1, 1],        # class B has zero weight -> excluded from this replicate's macro
                  [2, 0, 1, 1]], float)  # B again zero-weighted
    f1, sup = M.weighted_f1(W, T, PA, np.ones(4))
    assert np.isnan(f1[1, 1]) and np.isfinite(f1[0]).all()
    m, nex = M.macro_mean(f1)
    assert nex.tolist() == [0, 1, 1]
    assert m[1] == pytest.approx(np.nanmean(f1[1]))
    res = M.evaluate_contrast(W, T, PA, PF, np.ones(4))
    rows, reps = M.summarise_contrast(res, groups, n_reps=3)
    by = {r['class']: r for r in rows}
    assert by['B']['n_zero_support_reps'] == 2
    assert by['A']['n_zero_support_reps'] == 0
    assert by['MACRO']['n_zero_support_reps'] == 2
    assert by['C']['n_zero_support_reps'] == 0
    # with 3 replicates the 1 % limit is 0 -> per-class interval for B is not estimable
    assert by['B']['rer_ci_status'] in ('not_estimable_undefined_gt_1pct', 'undefined_point')


def test_point_zero_support_is_a_gate():
    groups = ['A', 'B']
    T = onehot(['A', 'A'], groups)
    res = M.evaluate_contrast(None, T, T, T, np.ones(2))
    with pytest.raises(M.GateError):
        M.summarise_contrast(res, groups, n_reps=1)


# ----------------------------------------------------------------------------------------------
# (e) RER undefined_point / not_estimable_undefined_gt_1pct status logic
# ----------------------------------------------------------------------------------------------

def test_rer_threshold():
    _, u = M.rer_values(1.0, 0.996)
    assert bool(u)
    r, u = M.rer_values(1.0, 0.994)
    assert not bool(u) and float(r) == pytest.approx(1.0)
    r, u = M.rer_values(np.nan, 0.5)
    assert np.isnan(r) and not bool(u)


def _reps(n_und, n_zero, n=5000):
    rer = np.linspace(-1, 1, n)
    und = np.zeros(n, bool)
    und[:n_und] = True
    zero = np.zeros(n, bool)
    zero[n_und:n_und + n_zero] = True
    rer = np.where(und | zero, np.nan, rer)
    return rer, und, zero


def test_rer_interval_status_rules():
    assert M.max_undefined_reps(5000) == 50
    r = M.rer_interval(True, *_reps(0, 0), is_macro=False)
    assert r['rer_ci_status'] == 'undefined_point' and np.isnan(r['rer_ci_low'])
    r = M.rer_interval(True, *_reps(0, 0), is_macro=True)
    assert r['rer_ci_status'] == 'undefined_point'
    r = M.rer_interval(False, *_reps(50, 0), is_macro=False)
    assert r['rer_ci_status'] == 'reported' and r['n_undefined_reps'] == 50 and np.isfinite(r['rer_ci_low'])
    r = M.rer_interval(False, *_reps(51, 0), is_macro=False)
    assert r['rer_ci_status'] == 'not_estimable_undefined_gt_1pct' and np.isnan(r['rer_ci_high'])
    r = M.rer_interval(False, *_reps(30, 21), is_macro=False)        # zero-support counts toward the 1 %
    assert r['rer_ci_status'] == 'not_estimable_undefined_gt_1pct'
    r = M.rer_interval(False, *_reps(400, 0), is_macro=True)          # macro: always reported with count
    assert r['rer_ci_status'] == 'reported' and r['n_undefined_reps'] == 400
    lo, hi = M.percentile_ci(_reps(400, 0)[0])
    assert (r['rer_ci_low'], r['rer_ci_high']) == (lo, hi)


def test_percentile_ci_linear():
    x = np.arange(101, dtype=float)
    lo, hi = M.percentile_ci(x)
    assert lo == pytest.approx(2.5) and hi == pytest.approx(97.5)
    assert all(np.isnan(v) for v in M.percentile_ci(np.array([np.nan])))


# ----------------------------------------------------------------------------------------------
# (f) reconciliation negative control
# ----------------------------------------------------------------------------------------------

GROUPS14 = [f'G{i:02d}' for i in range(14)]


def _hist_json(labels, n, rng):
    C = len(labels)
    cm = np.zeros((C, C), int)
    for i in range(C):
        p = np.where(np.arange(C) == i, 0.7, 0.3 / (C - 1))
        cm[i] = rng.multinomial(n // C + (1 if i < n % C else 0), p)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {'labels_order': list(labels), 'n_slides': int(cm.sum()), 'confusion_matrix': cm.tolist(),
            'config': {'top_k': 5, 'tie_break': 'generous_majority', 'dropped_labels': []},
            'point_estimates': {'macro_f1': float(f1.mean()),
                                'per_class': {c: {'f1': float(f1[i])} for i, c in enumerate(labels)}}}


@pytest.fixture()
def recon_fixture():
    rng = np.random.default_rng(7)
    full_labels = sorted(GROUPS14 + ['NORMAL'])
    jsons = {}
    for e in ('uni2', 'virchow2'):
        for m in M.METHODS:
            jsons[(e, m, 'full')] = _hist_json(full_labels, 2778, rng)
            jsons[(e, m, 'tumour')] = _hist_json(GROUPS14, 2604, rng)
    rows = []
    for e in ('uni2', 'virchow2'):
        for m in M.METHODS:
            pair = f'{M.ENC_S5[e]} + {M.METHOD_S5[m]}'
            fu, tu = jsons[(e, m, 'full')]['point_estimates'], jsons[(e, m, 'tumour')]['point_estimates']
            for c in GROUPS14:
                a, b = fu['per_class'][c]['f1'], tu['per_class'][c]['f1']
                rows.append({'Class': c, 'Pair': pair, 'Delta': (b - a) / (1 - a)})
            rows.append({'Class': 'Overall', 'Pair': pair,
                         'Delta': (tu['macro_f1'] - fu['macro_f1']) / (1 - fu['macro_f1'])})
    return pd.DataFrame(rows), jsons


def test_reconciliation_passes_on_consistent_fixture(recon_fixture):
    s5, jsons = recon_fixture
    rep = M.reconcile(s5, jsons, GROUPS14)
    assert rep['pass'] and rep['n_delta'] == 60 and rep['n_per_class_f1'] == 116 and rep['n_macro_f1'] == 8


def test_reconciliation_negative_controls(recon_fixture):
    s5, jsons = recon_fixture
    # perturb one confusion-matrix cell (count moved off the diagonal; total unchanged)
    bad = json.loads(json.dumps({str(k): v for k, v in jsons.items()}))
    pj = {k: bad[str(k)] for k in jsons}
    cm = pj[('virchow2', 'yottixel', 'tumour')]['confusion_matrix']
    cm[3][3] -= 1
    cm[3][4] += 1
    rep = M.reconcile(s5, pj, GROUPS14)
    assert not rep['pass']
    assert sum(not x['pass'] for x in rep['per_class_f1']) >= 1
    # perturb one submitted Delta cell
    s5b = s5.copy()
    s5b.loc[7, 'Delta'] += 1e-4
    rep = M.reconcile(s5b, jsons, GROUPS14)
    assert not rep['pass'] and sum(not x['pass'] for x in rep['delta_cells']) == 1
    # a missing S5 cell fails the exact-count requirement
    rep = M.reconcile(s5.iloc[1:], jsons, GROUPS14)
    assert not rep['pass'] and not rep['counts_ok']
    # a structural JSON mismatch fails (wrong tie-break)
    pj2 = {k: json.loads(json.dumps(v)) for k, v in jsons.items()}
    pj2[('uni2', 'ot_cosine', 'full')]['config']['tie_break'] = 'nearest'
    assert not M.reconcile(s5, pj2, GROUPS14)['pass']


# ----------------------------------------------------------------------------------------------
# (g) draw column-order assertion
# ----------------------------------------------------------------------------------------------

def test_check_draws_column_order():
    patients = ['P1', 'P2', 'P3']
    draws = np.array([[1, 1, 1], [3, 0, 0], [0, 2, 1], [1, 0, 2]], dtype=np.int64)
    ok = {'x': {'patient_draws_patients': patients}, 'y': {'patient_draws_patients': patients}}
    assert M.check_draws(draws, patients, ok, n_reps=4, n_patients=3)
    wrong = {'x': {'patient_draws_patients': patients}, 'y': {'patient_draws_patients': ['P2', 'P1', 'P3']}}
    with pytest.raises(M.GateError) as ex:
        M.check_draws(draws, patients, wrong, n_reps=4, n_patients=3)
    assert ex.value.gate == 'draws_binding'
    with pytest.raises(M.GateError):
        M.check_draws(draws, ['P2', 'P1', 'P3'], {}, n_reps=4, n_patients=3)       # unsorted population list
    with pytest.raises(M.GateError):
        M.check_draws(draws.astype(np.int32), patients, ok, n_reps=4, n_patients=3)
    with pytest.raises(M.GateError):
        M.check_draws(draws + np.eye(4, 3, dtype=np.int64), patients, ok, n_reps=4, n_patients=3)
    with pytest.raises(M.GateError):
        M.check_draws(draws, patients, {'x': {}}, n_reps=4, n_patients=3)


# ----------------------------------------------------------------------------------------------
# (h) non-return ID set assertion
# ----------------------------------------------------------------------------------------------

def test_check_nonreturn_sets():
    assert M.check_nonreturn_sets(dict(M.EXPECTED_NONRETURN), M.EXPECTED_NONRETURN)
    obs = dict(M.EXPECTED_NONRETURN)
    obs[('uni2', 'legacy_tumour_only', 'legacy')] = {16738, 99999}
    with pytest.raises(M.GateError) as ex:
        M.check_nonreturn_sets(obs, M.EXPECTED_NONRETURN)
    assert ex.value.gate == 'nonreturn_sets'
    assert list(ex.value.detail) == [str(('uni2', 'legacy_tumour_only', 'legacy'))]
    obs = dict(M.EXPECTED_NONRETURN)
    del obs[('hoptimus', 'query_only', '5.0')]
    with pytest.raises(M.GateError):
        M.check_nonreturn_sets(obs, M.EXPECTED_NONRETURN)
    obs = dict(M.EXPECTED_NONRETURN)
    obs[('hoptimus', 'all_tissue', '')] = {1}
    with pytest.raises(M.GateError):
        M.check_nonreturn_sets(obs, M.EXPECTED_NONRETURN)


# ----------------------------------------------------------------------------------------------
# (i) tie-block reversal
# ----------------------------------------------------------------------------------------------

def test_tie_reversal_changes_prediction_when_expected(votes):
    plurality, generous = votes
    labs = ['A', 'B', 'B', 'A', 'C', 'B', 'D', 'D', 'D', 'D']
    d = [.1, .2, .3, .4, .5, .5, .6, .7, .8, .9]
    assert M.has_boundary_tie(d)
    assert M.vote(labs, 'A', 'plurality', plurality, generous) == 'A'   # A-B 2:2, A closest
    rev, tie = M.reverse_boundary_block(labs, d)
    assert tie and rev[:6] == ['A', 'B', 'B', 'A', 'B', 'C'] and rev[6:] == labs[6:]
    assert M.vote(rev, 'A', 'plurality', plurality, generous) == 'B'    # changed by tie order only
    # generous with truth A still credits A only when A is tied at the top: here B now leads 3:2
    assert M.vote(rev, 'A', 'legacy_generous_ORACLE', plurality, generous) == 'B'


def test_tie_reversal_no_change_when_not_expected(votes):
    plurality, generous = votes
    labs = ['A', 'A', 'A', 'B', 'C', 'D', 'E', 'E', 'E', 'E']
    d = [.1, .2, .3, .4, .5, .5, .6, .7, .8, .9]
    rev, tie = M.reverse_boundary_block(labs, d)
    assert tie and rev[4] == 'D'
    assert M.vote(rev, 'A', 'plurality', plurality, generous) == M.vote(labs, 'A', 'plurality', plurality,
                                                                       generous) == 'A'
    # no boundary tie: list unchanged even if earlier ranks tie with each other
    d2 = [.1, .1, .3, .4, .5, .6, .7, .8, .9, 1.0]
    rev, tie = M.reverse_boundary_block(labs, d2)
    assert not tie and rev == labs
    # block spanning ranks 3..6 is reversed as a whole
    d3 = [.1, .2, .5, .5, .5, .5, .7, .8, .9, 1.0]
    rev, tie = M.reverse_boundary_block(list('ABCDEFGHIJ'), d3)
    assert tie and ''.join(rev) == 'ABFEDCGHIJ'


def test_tie_block_containment_failure():
    d = [.1, .2, .3, .4, .5, .5, .5, .5, .5, .5]     # tied block reaches the stored depth
    with pytest.raises(M.GateError) as ex:
        M.reverse_boundary_block(list('ABCDEFGHIJ'), d)
    assert ex.value.gate == 'tie_block_containment'


# ----------------------------------------------------------------------------------------------
# (j) hash mismatch -> FAILED.json
# ----------------------------------------------------------------------------------------------

def test_hash_inputs_records_mismatch_and_missing(tmp_path):
    good = tmp_path / 'good.txt'
    good.write_bytes(b'good')
    bad = tmp_path / 'bad.txt'
    bad.write_bytes(b'bad')
    pins = {'good': hashlib.sha256(b'good').hexdigest(), 'bad': hashlib.sha256(b'other').hexdigest(),
            'missing': '0' * 64}
    bindings, mism, data = M.hash_inputs({'good': str(good), 'bad': str(bad),
                                          'missing': str(tmp_path / 'nope')}, pins)
    assert set(mism) == {'bad', 'missing'}
    assert bindings['good']['match'] and bindings['good']['size'] == 4 and data['good'] == b'good'
    assert mism['missing']['error'].startswith('FileNotFoundError')


def test_main_hash_mismatch_writes_failed(tmp_path, monkeypatch, capsys):
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    (repo / 'scripts' / 'r2_3_internal_failure_tables.py').write_bytes(b'# not the frozen module\n')
    out_root = tmp_path / 'prosper_out'
    pop_dir = out_root / 'r2_1_mosaic_cpu_20261003' / 'attempt_001'
    pop_dir.mkdir(parents=True)
    (pop_dir / 'retrieval_population.csv').write_bytes(b'query_index,patient_id\n0,P1\n')
    monkeypatch.setenv('PROSPER_OUT', str(out_root))
    out = tmp_path / 'run'
    rc = M.main(['--out', str(out), '--repo', str(repo), '--comet-root', str(tmp_path / 'comet')])
    assert rc == 1
    assert not (out / 'COMPLETE.json').exists()
    failed = json.loads((out / 'FAILED.json').read_text())
    assert failed['status'] == 'FAILED' and failed['gate'] == 'input_hashes'
    assert {'population', 'vote_module'} <= set(failed['detail'])
    assert failed['detail']['population']['observed'] is not None            # present but wrong bytes
    assert failed['detail']['table_s5']['observed'] is None                  # missing
    assert failed['input_bindings']['script']['sha256'] == hashlib.sha256(open(SCRIPT, 'rb').read()).hexdigest()
    sums = (out / 'SHA256SUMS.txt').read_text()
    assert 'FAILED.json' in sums
    assert 'FAILED gate=input_hashes' in capsys.readouterr().out
    # a non-empty --out is refused without touching it
    rc = M.main(['--out', str(out), '--repo', str(repo), '--comet-root', str(tmp_path / 'comet')])
    assert rc == 2
