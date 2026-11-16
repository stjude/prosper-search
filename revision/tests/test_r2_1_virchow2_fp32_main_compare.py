"""Synthetic-fixture tests for scripts/r2_1_virchow2_fp32_main_compare.py (V2-main M5, specification candidate 002 + A1-A10).

Reuses the adapter tests' synthetic RR-5 world (frozen tests/test_r2_1_main_grid_plurality.py world + four
`virchow2_fp32/<m>` matrices, Run A and Run B option (ii)). No real data are read. Run with an explicit --basetemp
outside /tmp, `-p no:cacheprovider` and PYTHONDONTWRITEBYTECODE=1.
"""
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AT = _load('v2main_adapter_tests_world', os.path.join(REPO, 'tests', 'test_r2_1_virchow2_fp32_main_rr5_adapter.py'))
C = _load('v2main_compare_under_test', os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_compare.py'))
A, T = C.A, AT.T
METHODS = AT.METHODS


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('cmpworld'))
    v = AT.build_v2_world(root)
    AT.run_a(v)
    AT.run_b(v)
    v['root'] = root
    return v


def fresh_ns():
    ns = A.load_rr5()
    A.add_fp32_display(ns)
    return ns


def clone_run(v, name, drop_run_b=False):
    """Copy of the run dir (Run A, Run B, pins); full_spec paths still point at the original inputs."""
    dst = os.path.join(v['root'], name, 'run_001')
    shutil.copytree(v['run_dir'], dst, ignore=shutil.ignore_patterns('comparison_001*'))
    if drop_run_b:
        shutil.rmtree(os.path.join(dst, 'rr5_replacement_001'))
    return dst


def tamper_line(path, pred, col, fn):
    data = open(path, 'rb').read()
    head, lines = A.split_lines(data)
    hf = A.parse_fields(head)
    out, n = [], 0
    for ln in lines:
        f = A.parse_fields(ln)
        if n == 0 and pred(dict(zip(hf, f))):
            f[hf.index(col)] = fn(f[hf.index(col)])
            ln = A.format_fields(f)
            n += 1
        out.append(ln)
    assert n == 1
    os.chmod(path, 0o644)
    open(path, 'wb').write(A.join_lines(head, out))


def resum(d):
    """Rewrite d/SHA256SUMS.txt for the current bytes (simulates an internally consistent but wrong Run A so a
    later gate, not F9, is exercised)."""
    p = os.path.join(d, 'SHA256SUMS.txt')
    names = [ln.split('  ', 1)[1] for ln in open(p).read().splitlines()]
    os.chmod(p, 0o644)
    open(p, 'w').write(''.join(f'{A.sha256_file(os.path.join(d, n))}  {n}\n' for n in names))


@pytest.fixture(scope='module')
def compared(world):
    rep = C.do_compare(fresh_ns(), world['run_dir'], world['run001'])
    return rep, os.path.join(world['run_dir'], 'comparison_001')


# ----------------------------------------------------------------------------------------------
# full run (Run B present)
# ----------------------------------------------------------------------------------------------

def test_compare_passes_and_writes_m6_files(compared):
    rep, out = compared
    assert rep['pass'] and rep['a5']['pass'] and rep['routine_identity']['pass'] and rep['a3']['pass']
    want = {'point_estimates.csv', 'paired_differences.csv', 'neighbour_agreement_per_query.csv.gz',
            'neighbour_agreement_summary.csv', 'per_class_comparison.csv', 'materiality.json', 'product_changes.csv',
            'lineage_table_S3_virchow2.csv', 'lineage_table_S3_virchow2.footnotes.md', 'compare_report.json'}
    assert set(os.listdir(out)) == want
    assert rep['a5']['fields_compared'] > 0 and not rep['contract_draws']
    assert 'NOT draws #46' in rep['bootstrap_mode']
    # routine identity covers both encoders' aggregation - winner rows
    encs = {r['cell_id'].split('/')[0] for r in rep['routine_identity']['rows_checked']}
    assert encs == {'virchow2', 'uni2'} and all(r['equal'] for r in rep['routine_identity']['rows_checked'])


def test_paired_differences_and_materiality(compared, world):
    rep, out = compared
    pdf = pd.read_csv(os.path.join(out, 'paired_differences.csv'), dtype=str, keep_default_na=False)
    assert len(pdf) == len(METHODS) * len(T.KS) * (4 + 4)
    assert set(pdf['direction']) == {'fp32 - published'} and set(pdf['tie_order']) == {'canonical'}
    for _, r in pdf.iterrows():
        assert float(r['D_point']) == float(r['fp32_point']) - float(r['published_point'])
        assert r['n_reps'] == str(T.R_REPS)
    macro = pdf[pdf['metric'] == 'macro_f1']
    assert set(macro['flag_informational']) == {''} and set(macro['material_flag']) <= {'True', 'False'}
    other = pdf[pdf['metric'] != 'macro_f1']
    assert set(other['material_flag']) == {''} and set(other['flag_informational']) <= {'True', 'False'}
    assert set(other['flag_role']) == {'flag_informational'}
    hp = pdf[pdf['metric'].str.startswith(('hit', 'precision'))]
    assert set(hp['rule']) == {'rule_independent'}
    mj = json.load(open(os.path.join(out, 'materiality.json')))
    assert len(mj['cells']) == 32 and mj['n_package_cells'] == 16 and mj['n_informational_oracle_cells'] == 16
    assert mj['n_flag_informational_intervals'] == 96 and mj['n_flagged_rule_intervals_total'] == 128
    assert mj['package_flag'] == any(c['flag'] for c in mj['cells'] if c['rule'] == 'plurality')
    assert mj['rule_verbatim'] == C.MATERIALITY_RULE and 'disclosure trigger, not a hypothesis test' in \
        mj['chance_sentence']
    # paired rows recomputed independently from Run A per-cell reps are not available; the per-cell flag follows
    # the rule from the reported unrounded values
    for c in mj['cells']:
        lo, hi, d = float(c['lo']), float(c['hi']), c['delta']
        exp = (c['status'] == 'ok' and (lo > 0 or hi < 0) and abs(d) >= 0.005) or abs(d) >= 0.01
        assert c['flag'] == exp


class _S:
    def __init__(self, point, lo, hi, status='ok'):
        self.point, self.lo, self.hi, self.status = point, lo, hi, status


@pytest.mark.parametrize('s,flag', [
    (_S(0.006, 0.001, 0.011), True), (_S(0.004, 0.001, 0.007), False), (_S(-0.006, -0.011, -0.001), True),
    (_S(0.006, -0.001, 0.011), False), (_S(0.0100, -0.02, 0.03), True), (_S(-0.0101, -0.02, 0.03), True),
    (_S(0.009, 0.001, 0.02, 'too_few'), False), (_S(0.011, np.nan, np.nan, 'too_few'), True),
    (_S(0.005, 0.0001, 0.009), True), (_S(0.0049999, 0.0001, 0.009), False)])
def test_flag_rule(s, flag):
    assert C.flag_rule(s) is flag


def test_point_estimates_match_run_a_tables(compared, world):
    _, out = compared
    pe = pd.read_csv(os.path.join(out, 'point_estimates.csv'), dtype=str, keep_default_na=False)
    t3 = pd.read_csv(os.path.join(world['run_dir'], 'rr5_additive_001', 'table_s3_revised.csv'), dtype=str,
                     keep_default_na=False)
    r = pe[(pe['method'] == 'ot_cosine') & (pe['k'] == '5') & (pe['rule'] == 'plurality') &
           (pe['metric'] == 'macro_f1') & (pe['tie_order'] == 'canonical')].iloc[0]
    for enc, col in (('virchow2', 'published_value'), ('virchow2_fp32', 'fp32_value')):
        t = t3[(t3['cell_id'] == f'{enc}/ot_cosine') & (t3['k'] == '5') & (t3['rule'] == 'plurality') &
               (t3['metric'] == 'macro_f1') & (t3['tie_order'] == 'canonical')].iloc[0]
        assert r[col] == t['value']
    assert len(pe) == len(METHODS) * len(T.KS) * 2 * 8


def test_per_class_rows(compared):
    _, out = compared
    pc = pd.read_csv(os.path.join(out, 'per_class_comparison.csv'), dtype=str, keep_default_na=False)
    assert len(pc) == len(T.GROUPS) * len(METHODS) * len(T.KS) * 2
    assert set(pc['label']) == {'descriptive only; no multiplicity control'}


def test_neighbour_agreement_independent(compared, world):
    """Top-10 Jaccard / rank-1 recomputed from the test world's own masked matrices (stable mergesort)."""
    _, out = compared
    pq = pd.read_csv(os.path.join(out, 'neighbour_agreement_per_query.csv.gz'), dtype=str, keep_default_na=False)
    w = world['w']
    n = len(w['ids'])
    for m in METHODS:
        sub = pq[pq['method'] == m]
        assert len(sub) == n and list(sub['slide_id']) == [str(x) for x in w['ids']]
        tp = np.argsort(w['D'][f'virchow2/{m}'], axis=1, kind='mergesort')[:, :10]
        tf = np.argsort(w['D'][f'virchow2_fp32/{m}'], axis=1, kind='mergesort')[:, :10]
        jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(tp.tolist(), tf.tolist())]
        np.testing.assert_allclose(sub['jaccard_top10'].astype(float), jac, atol=5e-7)
        assert list(sub['rank1_agreement'].astype(int)) == list((tp[:, 0] == tf[:, 0]).astype(int))
        for k in T.KS:
            pp = w['pred'][f'virchow2/{m}']['canonical'][k]
            pf = w['pred'][f'virchow2_fp32/{m}']['canonical'][k]
            if 'top1' in pp and k == 1:
                pass
            assert set(sub[f'pred_agree_plurality_k{k}']) <= {'0', '1'}
    summ = pd.read_csv(os.path.join(out, 'neighbour_agreement_summary.csv'), dtype=str)
    assert {'mean', 'median', 'p5', 'fraction_eq_1', 'count_eq_1', 'n'} <= set(summ.columns)


def test_product_changes_independent(compared, world):
    """Virchow2 C32 slot winners: run_001 (virchow2/) versus Run B final (virchow2_fp32/), recomputed here."""
    _, out = compared
    ch = pd.read_csv(os.path.join(out, 'product_changes.csv'), dtype=str, keep_default_na=False)
    assert list(ch.columns) == list(C.PRODUCT_CHANGE_COLS)
    w0 = pd.read_csv(os.path.join(world['run001'], 'winners.csv'), dtype=str, keep_default_na=False)
    w1 = pd.read_csv(os.path.join(world['run_dir'], 'rr5_replacement_001', 'winners.csv'), dtype=str,
                     keep_default_na=False)
    exp = set()
    for (k, rule, order), g0 in w0[(w0.row_type == 'slot') & (w0.display_encoder == 'Virchow2')].groupby(
            ['k', 'rule', 'tie_order']):
        g1 = w1[(w1.row_type == 'slot') & (w1.display_encoder == 'Virchow2') & (w1.k == k) & (w1.rule == rule) &
                (w1.tie_order == order)]
        a = ';'.join(sorted(x.replace('virchow2/', 'virchow2_fp32/') for x in g0.winner_cell))
        b = ';'.join(sorted(g1.winner_cell))
        if a != b:
            exp.add(f'C32|Virchow2|{k}|{rule}|{order}')
    got = set(ch[ch.change_type == 'slot_winner']['key'])
    assert got == exp
    # every reported change differs
    assert all(r['before'] != r['after'] for _, r in ch.iterrows())
    # fig2b range values that involve Virchow2 change; all reported keys carry the relabelled scope
    rv = ch[ch.change_type == 'range_value']
    assert not any(k.endswith('|published_8') for k in rv['key'])


def test_lineage_table(compared):
    _, out = compared
    lt = pd.read_csv(os.path.join(out, 'lineage_table_S3_virchow2.csv'), dtype=str, keep_default_na=False)
    assert len(lt) == len(METHODS) * len(T.KS) * 8
    assert list(lt.columns) == list(C.LINEAGE_COLS)
    m = lt[lt.metric == 'macro_f1']
    assert set(m[m.rule == 'plurality'].flag_role) == {'package_flag_cell'}
    assert set(m[m.rule == 'oracle'].flag_role) == {'informational'}
    assert (lt.value_used_in_revised_manuscript == lt.fp32_value).all()
    assert set(lt[lt.metric != 'macro_f1'].material_flag) == {''}


# ----------------------------------------------------------------------------------------------
# ADDENDUM_001 (AMENDMENT_004): rr5_run_id_fp32 follows --run-dir; RUN_STAGE-consistent defaults
# ----------------------------------------------------------------------------------------------

PREVIOUS_RR5_RUN_ID_FP32 = 'r2_1_virchow2_fp32_main_20261013/run_001/rr5_additive_001'   # base l.86 constant


@pytest.mark.parametrize('stage', ['run_001', 'run_002'])
def test_addendum001_rr5_run_id_follows_run_dir(stage):
    got = C.rr5_run_id_fp32(os.path.join(A.ROOT, stage))
    assert got == f'r2_1_virchow2_fp32_main_20261013/{stage}/rr5_additive_001'
    assert C.rr5_run_id_fp32(os.path.join(A.ROOT, stage) + os.sep) == got
    alias = A.ROOT.replace('/PROSPER_DEPT/', '/PROSPER_DEPT/', 1)
    if alias != A.ROOT and os.path.isdir(os.path.dirname(os.path.dirname(alias))):
        assert C.rr5_run_id_fp32(os.path.join(alias, stage)) == got      # mount alias gives the same string


def test_addendum001_run_001_byte_identical_to_previous_constant():
    got = C.rr5_run_id_fp32(os.path.join(A.ROOT, 'run_001'))
    assert got.encode() == PREVIOUS_RR5_RUN_ID_FP32.encode()
    assert 'rr5_run_id_fp32' not in C.LINEAGE_CONST and 'rr5_run_id_fp32' in C.LINEAGE_COLS
    assert C.LINEAGE_CONST['rr5_run_id_published'] == 'r2_1_main_grid_plurality_20261012/run_001'   # unchanged


def test_addendum001_lineage_csv_follows_run_dir(world, compared, monkeypatch):
    _, out = compared
    lt = pd.read_csv(os.path.join(out, 'lineage_table_S3_virchow2.csv'), dtype=str, keep_default_na=False)
    assert set(lt.rr5_run_id_fp32) == {C.rr5_run_id_fp32(world['run_dir'])}
    monkeypatch.setattr(A, 'OUT', world['root'])
    want = {}
    for stage in ('run_001', 'run_002'):
        d = os.path.join(world['root'], 'addendum001_lineage', stage)
        shutil.copytree(world['run_dir'], d, ignore=shutil.ignore_patterns('comparison_001*'))
        assert C.main(['--run-dir', d, '--run001', world['run001']]) == 0
        got = pd.read_csv(os.path.join(d, 'comparison_001', 'lineage_table_S3_virchow2.csv'), dtype=str,
                          keep_default_na=False)
        assert set(got.rr5_run_id_fp32) == {f'addendum001_lineage/{stage}/rr5_additive_001'}
        want[stage] = got
    a, b = want['run_001'], want['run_002']
    assert list(a.columns) == list(C.LINEAGE_COLS)
    assert a.drop(columns='rr5_run_id_fp32').equals(b.drop(columns='rr5_run_id_fp32'))     # nothing else moves
    assert a.drop(columns='rr5_run_id_fp32').equals(lt.drop(columns='rr5_run_id_fp32'))


def test_addendum001_cli_defaults_use_run_stage(monkeypatch, capsys):
    S = _load('v2main_submit_for_compare_tests', os.path.join(REPO, 'scripts', 'r2_1_virchow2_fp32_main_submit.py'))
    assert A.RUN_STAGE == S.RUN_STAGE == 'run_003'
    want = os.path.abspath(os.path.join(A.ROOT, 'run_003'))
    seen = {}

    def fake_compare(ns, run_dir, *a, **k):
        seen['compare'] = run_dir
        return {'pass': True, 'package_flag': False}
    monkeypatch.setattr(C, 'do_compare', fake_compare)
    assert C.main([]) == 0
    assert seen['compare'] == want
    for cmd, fn, extra in (('eval-spec', 'cmd_eval_spec', ['--queue', 'q']), ('pins', 'cmd_pins', []),
                           ('run-a', 'cmd_run_a', []), ('run-b', 'cmd_run_b', ['--signoff', 'x'])):
        def fake(args, _c=cmd):
            seen[_c] = args.run_dir
            return 0
        monkeypatch.setattr(A, fn, fake)
        assert A.main([cmd] + extra) == 0
        assert os.path.abspath(seen[cmd]) == want
    capsys.readouterr()


# ----------------------------------------------------------------------------------------------
# STOP conditions
# ----------------------------------------------------------------------------------------------

def test_a5_stop_on_tampered_run_a_ci(world):
    d = clone_run(world, 'a5stop')
    p = os.path.join(d, 'rr5_additive_001', 'table_s3_revised.csv')
    tamper_line(p, lambda r: r['cell_id'] == 'virchow2_fp32/sampler' and r['metric'] == 'macro_f1' and
                r['ci_status'] == 'ok', 'ci_high', lambda x: f'{float(x) + 1e-6:.6f}')
    resum(os.path.join(d, 'rr5_additive_001'))
    with pytest.raises(C.CompareError, match='A5'):
        C.do_compare(fresh_ns(), d, world['run001'])
    rep = json.load(open(os.path.join(d, 'comparison_001', 'compare_report.json')))
    assert rep['pass'] is False and os.listdir(os.path.join(d, 'comparison_001')) == ['compare_report.json']


def test_routine_identity_stop(world):
    d = clone_run(world, 'ristop')
    r1 = os.path.join(world['root'], 'ristop', 'run001_copy')
    shutil.copytree(world['run001'], r1)
    tamper_line(os.path.join(r1, 'aggregation_contrasts.csv'), lambda r: r['encoder'] == 'uni2', 'ci_low',
                lambda x: f'{float(x) - 1e-6:.6f}')
    with pytest.raises(C.CompareError, match='routine identity'):
        C.do_compare(fresh_ns(), d, r1)


def test_a3_stop_on_tampered_run_b_winner(world):
    d = clone_run(world, 'a3stop')
    tamper_line(os.path.join(d, 'rr5_replacement_001', 'winners.csv'),
                lambda r: r['row_type'] == 'slot' and r['winner_cell'].startswith('virchow2_fp32/'), 'ci_high',
                lambda x: f'{float(x) + 1e-6:.6f}')
    with pytest.raises(C.CompareError, match='A3: winners.csv'):
        C.do_compare(fresh_ns(), d, world['run001'])


def test_a3_stop_on_tampered_fig2b_argmax(world):
    d = clone_run(world, 'a3stop2')
    tamper_line(os.path.join(d, 'rr5_replacement_001', 'fig2b_matrix.csv'),
                lambda r: r['row_type'] == 'range_across_encoders', 'argmax_cell', lambda x: 'uni2/sampler')
    with pytest.raises(C.CompareError, match='A3: fig2b_matrix.csv'):
        C.do_compare(fresh_ns(), d, world['run001'])


def test_a3_derived_one_unit_tolerance_and_strict(world, compared):
    """Derived columns recomputed from 6-dp inputs can differ from Run B's (unrounded-input) values by one 6-dp
    unit; the untampered world already shows such cases (reported, ambiguity 12). A one-unit tamper of a
    non-derived-equal row is tolerated and counted; --a3-strict turns every such case into STOP."""
    base = compared[0]['a3']
    d = clone_run(world, 'a3tol')
    p = os.path.join(d, 'rr5_replacement_001', 'fig2b_matrix.csv')
    rb = pd.read_csv(p, dtype=str, keep_default_na=False)
    rec, _ = C.a3_check(fresh_ns(), os.path.join(world['run_dir'], 'rr5_additive_001'),
                        os.path.join(world['run_dir'], 'rr5_replacement_001'), world['run001'])
    assert rec['pass'] and rec['fig2b_matrix.csv']['derived_fields_within_one_6dp_unit'] == \
        base['fig2b_matrix.csv']['derived_fields_within_one_6dp_unit']
    first = rb[(rb.row_type == 'range_across_aggregations') & rb.argmax_cell.str.startswith('virchow2_fp32/')].iloc[0]
    tamper_line(p, lambda r: all(r[c] == first[c] for c in rb.columns), 'value', lambda x: f'{float(x) + 1e-6:.6f}')
    rep = C.do_compare(fresh_ns(), d, world['run001'])
    n0 = base['fig2b_matrix.csv']['derived_fields_within_one_6dp_unit']
    assert rep['a3']['fig2b_matrix.csv']['derived_fields_within_one_6dp_unit'] in (n0 - 1, n0 + 1)
    tamper_line(p, lambda r: all(r[c] == first[c] for c in rb.columns if c != 'value'), 'value',
                lambda x: f'{float(x) + 3e-6:.6f}')
    d3 = os.path.join(d, 'comparison_001')
    shutil.rmtree(d3)
    with pytest.raises(C.CompareError, match='A3: fig2b_matrix.csv'):
        C.do_compare(fresh_ns(), d, world['run001'])
    d2 = clone_run(world, 'a3tol_strict')
    tamper_line(os.path.join(d2, 'rr5_replacement_001', 'fig2b_matrix.csv'),
                lambda r: r['row_type'] == 'range_across_aggregations' and r['argmax_cell'].startswith('virchow2_fp32/'),
                'value', lambda x: f'{float(x) + 1e-6:.6f}')
    with pytest.raises(C.CompareError, match='A3'):
        C.do_compare(fresh_ns(), d2, world['run001'], a3_strict=True)


def test_run_b_pending_mode(world):
    d = clone_run(world, 'pending', drop_run_b=True)
    with pytest.raises(C.CompareError, match='run-b-pending'):
        C.do_compare(fresh_ns(), d, world['run001'])
    rep = C.do_compare(fresh_ns(), d, world['run001'], run_b_pending=True)
    assert rep['a3']['status'] == 'pending' and rep['run_b'] == 'pending'
    ch = pd.read_csv(os.path.join(d, 'comparison_001', 'product_changes.csv'), dtype=str, keep_default_na=False)
    assert list(ch.change_type) == ['pending_run_b']
    d2 = clone_run(world, 'pending_contradiction')
    with pytest.raises(C.CompareError, match='exists'):
        C.do_compare(fresh_ns(), d2, world['run001'], run_b_pending=True)


def test_refuses_without_run_a_pass(world):
    d = clone_run(world, 'noruna')
    p = os.path.join(d, 'rr5_additive_001', 'evaluator_crosscheck.json')
    j = json.load(open(p))
    j['pass'] = False
    os.chmod(p, 0o644)
    open(p, 'w').write(json.dumps(j))
    with pytest.raises(C.CompareError, match='evaluator_crosscheck'):
        C.do_compare(fresh_ns(), d, world['run001'])


def test_never_overwrites(world, compared):
    with pytest.raises(A.AdapterError):
        C.do_compare(fresh_ns(), world['run_dir'], world['run001'])


def test_no_rr5_statistic_reimplemented():
    """A1/N.15 static check: the compare script calls RR-5's functions (contrast_stat, cell_statistics via the
    adapter, rank_cell, winners_and_panels, fig2b_rows, _csv_bytes) and defines none of them."""
    src = open(C.__file__).read()
    for fn in ('contrast_stat', 'cell_statistics', 'percentile_ci', 'rank_cell', 'winners_and_panels', 'fig2b_rows',
               'predict_cell', 'interval', '_csv_bytes', '_stat_cols'):
        assert f'def {fn}(' not in src
    for fn in ('ns.contrast_stat', 'ns.winners_and_panels', 'ns.fig2b_rows', 'ns._csv_bytes', 'ns._stat_cols',
               'A.recompute_cell_stats'):
        assert fn in src


# ----------------------------------------------------------------------------------------------
# repair candidate 002 regressions (implementation review 001: F3, F5, F6, F8, F9, F16, A3 tolerance)
# ----------------------------------------------------------------------------------------------

def test_f5_one_W_object_for_all_12_cells(world):
    d = clone_run(world, 'f5w')
    ns = fresh_ns()
    seen = []
    orig = ns.cell_statistics

    def spy(W, *a, **k):
        seen.append(W)
        return orig(W, *a, **k)
    ns.cell_statistics = spy
    rep = C.do_compare(ns, d, world['run001'])
    assert len(seen) == 12
    assert all(w is seen[0] for w in seen)
    assert rep['W_sha256'] == A.sha256_bytes(np.ascontiguousarray(seen[0]).tobytes())
    assert 'patient-order, not replicate-order' in rep['W_identity']
    assert 'not replicate-order' in C.__doc__


def test_f6_count_row_tamper_stops(world):
    d = clone_run(world, 'f6count')
    tamper_line(os.path.join(d, 'rr5_replacement_001', 'winners.csv'),
                lambda r: r['row_type'] == 'count' and r['slot_set'] == 'C32' and r['tie_order'] == 'canonical',
                'weight', lambda x: f'{float(x) + 1:.6f}')
    with pytest.raises(C.CompareError, match='A3: winners.csv:count'):
        C.do_compare(fresh_ns(), d, world['run001'])


def test_f6_count_rows_compared_strictly(compared):
    a3 = compared[0]['a3']
    c = a3['winners.csv:count']
    assert c['rows_recomputed'] == c['rows_run_b'] > 0 and c['derived_columns_tolerated'] == []


def test_f6_non_fp32_row_tamper_stops(world):
    d = clone_run(world, 'f6uni2')
    tamper_line(os.path.join(d, 'rr5_replacement_001', 'winners.csv'),
                lambda r: r['row_type'] == 'slot' and r['winner_cell'].startswith('uni2/'), 'ci_high',
                lambda x: f'{float(x) + 1e-6:.6f}')
    with pytest.raises(C.CompareError, match='F6: winners.csv'):
        C.do_compare(fresh_ns(), d, world['run001'])
    d2 = clone_run(world, 'f6uni2b')
    tamper_line(os.path.join(d2, 'rr5_replacement_001', 'fig2b_matrix.csv'),
                lambda r: r['row_type'] == 'cell' and r['cell_id'].startswith('uni2/'), 'value',
                lambda x: f'{float(x) + 1e-6:.6f}')
    with pytest.raises(C.CompareError, match='F6: fig2b_matrix.csv'):
        C.do_compare(fresh_ns(), d2, world['run001'])


def test_f6_non_fp32_identity_reported(compared):
    nf = compared[0]['a3']['non_fp32_identity']
    for name in ('winners.csv', 'fig2b_matrix.csv'):
        assert nf[name]['identical'] and nf[name]['run001_rows_compared'] == nf[name]['run_b_rows_compared'] > 0
        # removed counts may differ only through range_across_encoders rows whose argmax/argmin is Virchow2 in
        # one run and not the other; those rows are F6_DEPENDENT and recomputed by A3
        assert nf[name]['run001_virchow2_rows_removed'] > 0 and nf[name]['run_b_virchow2_fp32_rows_removed'] > 0
    assert compared[0]['a3']['argmax_k_fp32_vs_run_a']['identical']


def test_f6_argmax_k_fp32_tamper_stops(world):
    d = clone_run(world, 'f6argk')
    tamper_line(os.path.join(d, 'rr5_replacement_001', 'winners.csv'),
                lambda r: r['row_type'] == 'argmax_k' and r['winner_cell'].startswith('virchow2_fp32/'), 'value',
                lambda x: f'{float(x) + 1e-6:.6f}')
    with pytest.raises(C.CompareError, match='argmax_k'):
        C.do_compare(fresh_ns(), d, world['run001'])


def test_f3_both_published_scopes_relabelled(world, compared):
    rb = os.path.join(world['run_dir'], 'rr5_replacement_001')
    f2b = pd.read_csv(os.path.join(rb, 'fig2b_matrix.csv'), keep_default_na=False, dtype=str)
    scopes = set(f2b['scope']) - {''}
    assert set(A.A8_SCOPE_MAP.values()) <= scopes and not scopes & set(A.A8_SCOPE_MAP)
    assert not any(s.startswith('published_8') for s in scopes)
    ch = pd.read_csv(os.path.join(compared[1], 'product_changes.csv'), dtype=str, keep_default_na=False)
    assert not ch['key'].str.contains('published_8').any()
    a3 = compared[0]['a3']['fig2b_matrix.csv']
    assert a3['rows_recomputed'] == a3['rows_run_b']


def test_f9_run_a_sums_verified_before_compare(world):
    d = clone_run(world, 'f9')
    p = os.path.join(d, 'rr5_additive_001', 'hit_precision.csv')
    os.chmod(p, 0o644)
    data = open(p, 'rb').read()
    open(p, 'wb').write(data.replace(b'0.', b'1.', 1))
    with pytest.raises(C.CompareError, match='F9'):
        C.do_compare(fresh_ns(), d, world['run001'])
    assert not os.path.exists(os.path.join(d, 'comparison_001'))
    d2 = clone_run(world, 'f9b')
    s = os.path.join(d2, 'rr5_additive_001', 'SHA256SUMS.txt')
    lines = [ln for ln in open(s).read().splitlines() if not ln.endswith('  table_s3_revised.csv')]
    os.chmod(s, 0o644)
    open(s, 'w').write('\n'.join(lines) + '\n')
    with pytest.raises(C.CompareError, match='not listed'):
        C.do_compare(fresh_ns(), d2, world['run001'])


def test_f16_six_dp_tie_documented(compared):
    rep = compared[0]
    assert 'fail-safe' in rep['a3']['six_dp_tie'] and 'never a false pass' in rep['a3']['six_dp_tie']
    assert rep['a3']['tolerance'] == C.A3_TOLERANCE_NOTE


def test_f1_compare_report_binds_run_b_faithfulness(world, compared):
    rep = compared[0]
    fr = os.path.join(world['run_dir'], 'rr5_replacement_001', 'faithfulness_report.json')
    assert rep['run_b_faithfulness_report']['sha256'] == A.sha256_file(fr)
    assert rep['run_a_sha256sums']['lines_verified'] > 0


def test_cli_refuses_a3_strict_and_rr5_override(world, tmp_path, capsys):
    assert C.main(['--run-dir', str(tmp_path / 'none'), '--a3-strict']) == 1
    assert '--a3-strict is refused' in capsys.readouterr().err
    other = tmp_path / 'rr5_copy.py'
    shutil.copy(A.RR5_PATH, other)
    assert C.main(['--run-dir', str(tmp_path / 'none'), '--rr5', str(other), '--rr5-sha', A.RR5_SHA]) == 1
    assert 'F8' in capsys.readouterr().err
    assert C.main(['--run-dir', str(tmp_path / 'none'), '--rr5-sha', '0' * 64]) == 1
    assert 'F8' in capsys.readouterr().err
    assert not os.path.exists(tmp_path / 'none')
