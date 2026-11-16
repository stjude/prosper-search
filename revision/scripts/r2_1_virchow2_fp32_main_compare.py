#!/usr/bin/env python3
"""V2-main M5: row-by-row comparison of the fp32 Virchow2 cells with the published-matrix Virchow2 cells.

Specification: ROOT/contract_candidate_002/SPECIFICATION.md (M5, M6) with LEAD_RESOLUTIONS_002.md and
LEAD_RESOLUTIONS_002_ADDENDUM.md (A1, A3, A5, A6). Output: ROOT/<RUN_STAGE>/comparison_001/ (default --run-dir).

Every statistic comes from RR-5's own functions called through the adapter's hash-checked namespace (A1/N.15):
`cell_statistics` (via the adapter's recompute_cell_stats, one load of draws #46 and one population ordering, A5),
`contrast_stat`, `rank_cell` (canonical order), `winners_and_panels`, `fig2b_rows` and the row writer
`_csv_bytes`/`_stat_cols`. This script only selects, compares and tabulates. Gates (any failure = STOP, exit 1,
`compare_report.json` with pass=false and no comparison products):
  - Run A COMPLETE with passing bit_identity_report.json and evaluator_crosscheck.json;
  - F9: every line of Run A rr5_additive_001/SHA256SUMS.txt verifies (sha256sum -c equivalent) before any read;
  - A5: every recomputed Stat (value, ci_low, ci_high, ci_status as formatted by RR-5's writer) equals the Run A
    table_s3_revised.csv / hit_precision.csv row of the same cell/k/rule/metric/order. A5 detects patient-order,
    not replicate-order, misalignment of W (F5); replicate-order alignment holds by construction (one W object is
    passed to cell_statistics for all 12 cells, tested; sha256(W.tobytes()) is recorded once in compare_report);
  - M5.2 routine identity: every run_001 aggregation_contrasts.csv row of virchow2 and uni2 is reproduced byte for
    byte (and, on the specification draws, the quoted lines 23 and 20);
  - A3 (only with a COMPLETE Run B): the Virchow2 C32 slot winners, every winners.csv count row (F6, strict) and
    every fig2b range row recomputed by RR-5's functions from the Run A table_s3 6-dp values equal the
    post-row-drop Run B products; the virchow2_fp32 argmax_k rows equal Run A's byte for byte; and (F6, A4-style)
    Run B winners.csv / fig2b_matrix.csv without virchow2_fp32/ rows equal run_001 without virchow2/ rows byte for
    byte, except the rows that legitimately depend on the Virchow2 cells and are themselves recomputed by A3
    (winners count rows other than tie_order submitted_xlsx; fig2b range_across_encoders rows).
    Only `gap_to_runner_up` (winners) and `value` (fig2b ranges) may differ by one unit in the 6th decimal; the
    `a3_strict` function argument (tests only; refused on the CLI) removes even that tolerance.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(HERE, 'r2_1_virchow2_fp32_main_rr5_adapter.py')


def _load_adapter(path=ADAPTER_PATH):
    spec = importlib.util.spec_from_file_location('r2_1_v2main_rr5_adapter_for_compare', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load_adapter()
FP32, PUB = A.FP32, A.PUB
CompareError = A.AdapterError

BOOTSTRAP_MODE = 'whole-patient multiplicity weights, n_reps = 5000, draws #46'
DIRECTION = 'fp32 - published'
F1_METRICS = ('macro_f1', 'micro_f1')
HP_METRICS = ('hit_micro', 'hit_macro', 'precision_micro', 'precision_macro')
JACCARD_DEPTHS = (1, 3, 5, 10)
ROUTINE_ENCODERS = ('virchow2', 'uni2')
# (cell_id, winner_cell): (1-based line incl. header, value, ci_low, ci_high) as quoted in specification M5.2
ROUTINE_QUOTED = {('virchow2/median_pooling', 'virchow2/yottixel'): (23, '-0.035223', '-0.055391', '-0.015462'),
                  ('uni2/median_pooling', 'uni2/ot_cosine'): (20, '-0.043189', '-0.060816', '-0.026835')}
MATERIALITY_RULE = (
    "A cell is *flagged* if, with \u0394 = `D.point` (unrounded) and [`D.lo`, `D.hi`] from M5.2: "
    "(`D.status == 'ok'` and (`D.lo > 0` or `D.hi < 0`) and |\u0394| \u2265 0.005) **or** |\u0394| \u2265 0.01. "
    "If `D.status != 'ok'`, only the |\u0394| \u2265 0.01 arm applies, and the status is reported. "
    "The **package materiality flag** = any flagged cell among the **16 plurality macro-F1 cells** "
    "(4 methods \u00d7 k \u2208 {1,3,5,10}, canonical order). The 16 oracle macro-F1 cells receive the same per-cell "
    "flag and are reported as \"informational\". They do not set the package flag.")
A6_RULE = ("Micro-F1 (both rules) and hit@k/precision@k (micro and macro, both rules) receive the same per-cell rule "
           "as an informational-only column `flag_informational`. It does not set the package flag. Per-class "
           "metrics and neighbour agreement are reported descriptively without a flag.")
OUTCOME_NOT_SET = 'numerically consistent with the submitted (published-bank) Virchow2 values'
OUTCOME_SET = ('flagged cells are listed with \u0394, direction and interval; every manuscript statement that uses '
               'Virchow2 then needs re-audit: winners, encoder ordering, Fig. 2C and S-R2.1-E')
DISCLOSURE = 'The flag is a disclosure trigger, not a hypothesis test.'
RULE_DEFINITION = {'plurality': 'plurality: nearest-ranked tie-break', 'oracle': 'oracle: legacy generous',
                   'rule_independent': 'rule-independent (hit@k / precision@k)'}
LINEAGE_CONST = {
    'submitted_ci_method': 'pps_cluster_one_slide_class_stratified (submitted)',
    'revised_ci_method': 'whole-patient bootstrap, 5,000 draws, seed 123',
    'submitted_bank': 'published: 2,807 float16 + 114 float32 stored',
    'revised_bank': 'fp32 regeneration, all 2,921 float32 stored',
    'rr5_run_id_published': 'r2_1_main_grid_plurality_20261012/run_001',
    'author_decision': ('2026-10-13 16:1x CDT decision B: replace submitted Virchow2 rows with fp32 rows; '
                        'ledger entry'),
    'bank_identity_published': ('TMPF per-file binding #29 (stage0 inventory.csv sha256 '
                                '9449de4872eac5a19c2479b8b829f790ac19c9d87f52a9af1b00100c1ebd261e)'),
    'bank_identity_fp32': ('FP32 binding #30 (reconcile_v18.json sha256 '
                           '83100633fb109c9bfda15b235d3968ed2e86f85e9419a3161cf396eee15ba828)'),
}
LINEAGE_FOOTNOTES = """# Footnotes to lineage_table_S3_virchow2.csv (contract M6)

1. The submitted (published) Virchow2 bank stores 2,807 slides as float16 and 114 as float32; by author decision B
   (2026-10-13) the fp32 rows replace the submitted Virchow2 main-grid rows.
2. The published Virchow2 tumour-only rows (2,604/14 scope) are not regenerated by this package and remain on the
   published bank until their separate package; until then the revised manuscript mixes fp32 main-grid Virchow2
   rows with published-bank tumour-only rows.
3. The materiality flag is a disclosure trigger, not a hypothesis test.
"""
PRODUCT_CHANGE_COLS = ('product', 'change_type', 'key', 'before', 'after', 'before_value', 'after_value')
A3_DERIVED = {'winners.csv': ('gap_to_runner_up',), 'fig2b_matrix.csv': ('value',)}
A3_TOL = 1e-6 + 1e-12      # one unit in RR-5's 6th decimal (rounding of a difference of 6-dp values); see report
A3_TOLERANCE_NOTE = ('+/-1 unit in the 6th decimal only for winners.gap_to_runner_up and fig2b range value: RR-5 rounds '
                     'the difference of unrounded values, A3 the difference of 6-dp values; they differ by 0 or exactly '
                     '1 unit, never 2 (LEAD_IMPL_DECISIONS_001). Every categorical/copied field is compared strictly.')
SIX_DP_TIE_NOTE = ('F16: RR-5 picks winners by exact equality of unrounded points; A3 recomputes from 6-dp writer '
                   'values, so a near-tie below the 6th decimal becomes a 6-dp tie and STOPs labelled six_dp_tie. '
                   'This is fail-safe (a false STOP, never a false pass); a six_dp_tie STOP needs lead inspection, '
                   'not a tolerance.')
A3_COUNT_SKIP_ORDER = 'submitted_xlsx'     # count rows from the xlsx (not recomputable from Run A values)
F6_DEPENDENT = {'winners.csv': lambda r: r['row_type'] == 'count' and r['tie_order'] != A3_COUNT_SKIP_ORDER,
                'fig2b_matrix.csv': lambda r: r['row_type'] == 'range_across_encoders'}


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------

def _csv(rows, cols):
    df = pd.DataFrame(list(rows), columns=list(cols))
    return df.to_csv(index=False, lineterminator='\n').encode()


def _r(x):
    return repr(float(x))


def stat_fields(ns, s):
    """value/ci_low/ci_high/ci_status exactly as RR-5's writer formats them (_stat_cols -> _csv_bytes)."""
    _, hf, rows = A.table(ns._csv_bytes(pd.DataFrame([ns._stat_cols(s)])))
    return dict(zip(hf, rows[0][1]))


def records(data):
    _, hf, rows = A.table(data)
    return hf, [dict(zip(hf, f)) for _, f in rows]


def flag_rule(D):
    """Pre-registered materiality rule (specification M5.5), on the unrounded D.point / D.lo / D.hi."""
    d = float(D.point)
    ok = D.status == 'ok' and np.isfinite(D.lo) and np.isfinite(D.hi)
    excl = bool(ok and (D.lo > 0 or D.hi < 0))
    return bool((excl and abs(d) >= 0.005) or abs(d) >= 0.01)


def _q(st, order, k, rule, metric):
    if rule == 'rule_independent':
        return st[(order, k, 'hp')][metric]
    return st[(order, k, rule)][metric]


def _metric_rules():
    for rule in ('plurality', 'oracle'):
        for m in F1_METRICS:
            yield rule, m
    for m in HP_METRICS:
        yield 'rule_independent', m


# ----------------------------------------------------------------------------------------------
# preconditions
# ----------------------------------------------------------------------------------------------

def require_run_a(run_dir):
    ra = os.path.join(run_dir, 'rr5_additive_001')
    if os.path.exists(os.path.join(ra, 'FAILED.json')):
        raise CompareError('Run A has FAILED.json')
    st = json.load(open(os.path.join(ra, 'COMPLETE.json')))
    if st.get('status') != 'COMPLETE':
        raise CompareError('Run A is not COMPLETE')
    for f in ('bit_identity_report.json', 'evaluator_crosscheck.json'):
        if not json.load(open(os.path.join(ra, f))).get('pass'):
            raise CompareError(f'Run A {f} did not pass')
    sums = A.verify_sha256sums(ra)
    spec = json.load(open(os.path.join(run_dir, 'rr5_additive_001.full_spec.json')))['full_spec']
    return ra, spec, sums


def run_b_state(run_dir, pending):
    rb = os.path.join(run_dir, 'rr5_replacement_001')
    fr = os.path.join(rb, 'faithfulness_report.json')
    present = os.path.exists(fr)
    if pending:
        if os.path.exists(rb):
            raise CompareError('--run-b-pending given but rr5_replacement_001 exists')
        return None
    if not present:
        raise CompareError('Run B faithfulness_report.json absent; pass --run-b-pending only if Run B is infeasible '
                           'or not released (A3: cross-cell replacement products are then reported pending)')
    if not json.load(open(fr)).get('pass'):
        raise CompareError('Run B faithfulness_report.json pass != true')
    if os.path.exists(os.path.join(rb, 'FAILED.json')):
        raise CompareError('Run B has FAILED.json')
    return rb


# ----------------------------------------------------------------------------------------------
# A5 and routine identity
# ----------------------------------------------------------------------------------------------

def a5_check(ns, stats, run_a, k_values, orders):
    """Recomputed Stats vs Run A table_s3_revised.csv and hit_precision.csv, at RR-5's writer formatting."""
    fails, n = [], 0
    _, t3 = records(A.read_product(run_a, 'table_s3_revised.csv'))
    _, hp = records(A.read_product(run_a, 'hit_precision.csv'))
    i3 = {(r['cell_id'], r['k'], r['rule'], r['metric'], r['tie_order']): r for r in t3}
    ih = {(r['cell_id'], r['k'], r['metric'], r['tie_order']): r for r in hp}
    for cid, st in stats.items():
        n3 = sum(1 for r in t3 if r['cell_id'] == cid)
        nh = sum(1 for r in hp if r['cell_id'] == cid)
        c3 = ch = 0
        for order in orders:
            for k in k_values:
                for rule in ('plurality', 'oracle'):
                    for m, s in st[(order, k, rule)].items():
                        if m == 'per_class':
                            continue
                        row = i3.get((cid, str(k), rule, m, order))
                        c3 += 1
                        if row is None:
                            fails.append(f'A5: no Run A table_s3 row {cid} k={k} {rule} {m} {order}')
                            continue
                        f = stat_fields(ns, s)
                        for col in ('value', 'ci_low', 'ci_high', 'ci_status'):
                            n += 1
                            if f[col] != row[col]:
                                fails.append(f'A5: {cid} k={k} {rule} {m} {order} {col}: recomputed {f[col]!r} '
                                             f'!= Run A {row[col]!r}')
                for m, s in st[(order, k, 'hp')].items():
                    row = ih.get((cid, str(k), m, order))
                    ch += 1
                    if row is None:
                        fails.append(f'A5: no Run A hit_precision row {cid} k={k} {m} {order}')
                        continue
                    f = stat_fields(ns, s)
                    for col in ('value', 'ci_low', 'ci_high', 'ci_status'):
                        n += 1
                        if f[col] != row[col]:
                            fails.append(f'A5: {cid} k={k} {m} {order} {col}: recomputed {f[col]!r} != Run A '
                                         f'{row[col]!r}')
        if c3 != n3 or ch != nh:
            fails.append(f'A5: {cid}: {c3}/{n3} table_s3 and {ch}/{nh} hit_precision rows covered')
    return {'fields_compared': n, 'cells': sorted(stats), 'pass': not fails, 'failures': fails[:200]}, fails


def routine_identity(ns, stats, run001, contract_draws):
    """M5.2 routine-identity gate on run_001 aggregation_contrasts.csv (header = line 1)."""
    data = A.read_product(run001, 'aggregation_contrasts.csv')
    hf, rows = records(data)
    fails, checked = [], []
    per_enc = {e: 0 for e in ROUTINE_ENCODERS}
    for ln, r in enumerate(rows, start=2):
        enc = r['encoder']
        if enc not in ROUTINE_ENCODERS:
            continue
        a, w = r['cell_id'], r['winner_cell']
        if a not in stats or w not in stats:
            fails.append(f'routine identity: Stats for {a} / {w} not recomputed')
            continue
        k, rule, order, metric = int(r['k']), r['rule'], r['tie_order'], r['metric']
        s = ns.contrast_stat(stats[a][(order, k, rule)][metric], stats[w][(order, k, rule)][metric])
        f = stat_fields(ns, s)
        same = all(f[c] == r[c] for c in ('value', 'ci_low', 'ci_high', 'ci_status'))
        per_enc[enc] += 1
        checked.append({'line': ln, 'cell_id': a, 'winner_cell': w, 'k': k, 'rule': rule, 'tie_order': order,
                        'run001': [r[c] for c in ('value', 'ci_low', 'ci_high', 'ci_status')],
                        'recomputed': [f[c] for c in ('value', 'ci_low', 'ci_high', 'ci_status')], 'equal': same})
        if not same:
            fails.append(f'routine identity: line {ln} {a} - {w}: recomputed {f} != run_001')
    for e, n in per_enc.items():
        if n == 0:
            fails.append(f'routine identity: no run_001 aggregation_contrasts row for {e}')
    quoted = {}
    if contract_draws:
        for (a, w), (ln, v, lo, hi) in ROUTINE_QUOTED.items():
            hit = [c for c in checked if c['cell_id'] == a and c['winner_cell'] == w and c['k'] == 5
                   and c['rule'] == 'plurality' and c['tie_order'] == 'canonical']
            ok = len(hit) == 1 and hit[0]['line'] == ln and hit[0]['recomputed'][:3] == [v, lo, hi] and hit[0]['equal']
            quoted[f'{a} - {w}'] = {'expected_line': ln, 'expected': [v, lo, hi], 'ok': ok}
            if not ok:
                fails.append(f'routine identity: quoted row {a} - {w} (line {ln}, {v} [{lo}, {hi}]) not reproduced')
    return {'rows_checked': checked, 'quoted_rows': quoted, 'contract_draws': contract_draws,
            'pass': not fails, 'failures': fails}, fails


# ----------------------------------------------------------------------------------------------
# M5.1, M5.2, M5.4, M5.5
# ----------------------------------------------------------------------------------------------

def point_estimates(ns, stats, methods, k_values, orders):
    out = []
    for m in methods:
        sp, sf = stats[f'{PUB}/{m}'], stats[f'{FP32}/{m}']
        for k in k_values:
            for order in orders:
                for rule, metric in _metric_rules():
                    a, b = _q(sp, order, k, rule, metric), _q(sf, order, k, rule, metric)
                    fa, fb = stat_fields(ns, a), stat_fields(ns, b)
                    out.append({'method': m, 'k': k, 'tie_order': order, 'rule': rule, 'metric': metric,
                                'published_cell': f'{PUB}/{m}', 'fp32_cell': f'{FP32}/{m}',
                                'published_value': fa['value'], 'published_ci_low': fa['ci_low'],
                                'published_ci_high': fa['ci_high'], 'published_ci_status': fa['ci_status'],
                                'fp32_value': fb['value'], 'fp32_ci_low': fb['ci_low'], 'fp32_ci_high': fb['ci_high'],
                                'fp32_ci_status': fb['ci_status'], 'published_value_unrounded': _r(a.point),
                                'fp32_value_unrounded': _r(b.point),
                                'delta_fp32_minus_published_unrounded': _r(b.point - a.point)})
    return out


PAIRED_COLS = ('method', 'k', 'rule', 'tie_order', 'metric', 'fp32_cell', 'published_cell', 'direction',
               'fp32_point', 'published_point', 'D_point', 'D_lo', 'D_hi', 'D_status', 'D_value_6dp',
               'D_ci_low_6dp', 'D_ci_high_6dp', 'n_invalid_replicates', 'n_reps', 'bootstrap_mode',
               'flag_role', 'material_flag', 'flag_informational')


def paired_differences(ns, stats, methods, k_values, mode):
    rows, mat = [], []
    for m in methods:
        sp, sf = stats[f'{PUB}/{m}'], stats[f'{FP32}/{m}']
        for k in k_values:
            for rule, metric in _metric_rules():
                a, b = _q(sf, 'canonical', k, rule, metric), _q(sp, 'canonical', k, rule, metric)
                D = ns.contrast_stat(a, b)
                reps = np.asarray(D.reps, float)
                f = stat_fields(ns, D)
                flag = flag_rule(D)
                if metric == 'macro_f1':
                    role = 'package_flag_cell' if rule == 'plurality' else 'informational'
                else:
                    role = 'flag_informational'
                row = {'method': m, 'k': k, 'rule': rule, 'tie_order': 'canonical', 'metric': metric,
                       'fp32_cell': f'{FP32}/{m}', 'published_cell': f'{PUB}/{m}', 'direction': DIRECTION,
                       'fp32_point': _r(a.point), 'published_point': _r(b.point), 'D_point': _r(D.point),
                       'D_lo': _r(D.lo), 'D_hi': _r(D.hi), 'D_status': D.status, 'D_value_6dp': f['value'],
                       'D_ci_low_6dp': f['ci_low'], 'D_ci_high_6dp': f['ci_high'],
                       'n_invalid_replicates': int((~np.isfinite(reps)).sum()), 'n_reps': int(reps.size),
                       'bootstrap_mode': mode, 'flag_role': role,
                       'material_flag': str(flag) if metric == 'macro_f1' else '',
                       'flag_informational': str(flag) if metric != 'macro_f1' else ''}
                rows.append(row)
                if metric == 'macro_f1':
                    mat.append({'cell': f'{FP32}/{m}', 'published_cell': f'{PUB}/{m}', 'method': m, 'k': k,
                                'rule': rule, 'tie_order': 'canonical', 'delta': float(D.point),
                                'delta_repr': _r(D.point), 'lo': _r(D.lo), 'hi': _r(D.hi), 'status': D.status,
                                'flag': flag, 'role': role, 'direction': ('fp32 higher' if D.point > 0 else
                                                                         'fp32 lower' if D.point < 0 else 'equal')})
    return rows, mat


def materiality(mat, paired, mode, n_reps):
    pkg = [c for c in mat if c['role'] == 'package_flag_cell']
    inf = [c for c in mat if c['role'] == 'informational']
    a6 = [r for r in paired if r['flag_role'] == 'flag_informational']
    if len(pkg) != 16 or len(inf) != 16:
        raise CompareError(f'materiality: {len(pkg)} package and {len(inf)} oracle cells, expected 16 + 16')
    package_flag = any(c['flag'] for c in pkg)
    n_total = len(pkg) + len(inf) + len(a6)
    chance = (f'With {len(pkg)} package-flag intervals ({len(pkg) + len(inf)} including the informational oracle '
              f'macro-F1 cells; {n_total} flagged-rule intervals in total including the {len(a6)} A6 '
              f'flag_informational intervals for micro-F1, hit@k and precision@k), some may exclude 0 by chance even '
              f'under no systematic change; the |\u0394| \u2265 0.005 floor limits this. {DISCLOSURE}')
    flagged = [c for c in mat if c['flag']]
    return {'rule_verbatim': MATERIALITY_RULE, 'a6_rule_verbatim': A6_RULE, 'direction': DIRECTION,
            'bootstrap_mode': mode, 'n_reps': n_reps, 'cells': mat, 'package_flag': package_flag,
            'n_package_cells': len(pkg), 'n_informational_oracle_cells': len(inf),
            'n_flag_informational_intervals': len(a6), 'n_flagged_rule_intervals_total': n_total,
            'n_package_cells_flagged': sum(c['flag'] for c in pkg),
            'n_informational_oracle_cells_flagged': sum(c['flag'] for c in inf),
            'n_flag_informational_flagged': sum(r['flag_informational'] == 'True' for r in a6),
            'flagged_cells': flagged,
            'outcome_wording': OUTCOME_SET if package_flag else OUTCOME_NOT_SET,
            'replacement': 'In both cases, the fp32 rows replace the submitted rows (decision B).',
            'chance_sentence': chance, 'disclosure': DISCLOSURE}


def per_class(ns, stats, methods, k_values, labels_order):
    out = []
    for m in methods:
        sp, sf = stats[f'{PUB}/{m}'], stats[f'{FP32}/{m}']
        for k in k_values:
            for rule in ('plurality', 'oracle'):
                pp, pf = sp[('canonical', k, rule)]['per_class'], sf[('canonical', k, rule)]['per_class']
                for g in pp:
                    a, b = pf[g]['f1'], pp[g]['f1']
                    D = ns.contrast_stat(a, b)
                    f = stat_fields(ns, D)
                    sup_p, sup_f = pp[g]['support'].point, pf[g]['support'].point
                    if sup_p != sup_f:
                        raise CompareError(f'per-class support differs for {m} {labels_order[g]}')
                    out.append({'method': m, 'k': k, 'rule': rule, 'tie_order': 'canonical',
                                'group': labels_order[g], 'published_f1': _r(b.point), 'fp32_f1': _r(a.point),
                                'delta_fp32_minus_published': _r(D.point), 'D_lo': _r(D.lo), 'D_hi': _r(D.hi),
                                'D_status': D.status, 'D_value_6dp': f['value'], 'D_ci_low_6dp': f['ci_low'],
                                'D_ci_high_6dp': f['ci_high'], 'n_support': _r(sup_p),
                                'label': 'descriptive only; no multiplicity control'})
    return out


# ----------------------------------------------------------------------------------------------
# M5.3 neighbour agreement
# ----------------------------------------------------------------------------------------------

def neighbour_agreement(keep, popd, run_a, methods, k_values):
    pq = pd.read_csv(os.path.join(run_a, 'per_query_predictions.csv.gz'), dtype=str, keep_default_na=False)
    per, summ = [], []
    n = int(popd['n'])
    for m in methods:
        tp, tf = keep[f'{PUB}/{m}']['top']['canonical'], keep[f'{FP32}/{m}']['top']['canonical']
        if tp.shape != (n, 10) or tf.shape != (n, 10):
            raise CompareError(f'neighbour agreement: top-10 shape {tp.shape}/{tf.shape} for {m}')
        cols = {}
        for j in JACCARD_DEPTHS:
            v = np.empty(n)
            for i in range(n):
                a, b = set(tp[i, :j].tolist()), set(tf[i, :j].tolist())
                v[i] = len(a & b) / len(a | b)
            cols[f'jaccard_top{j}'] = v
        cols['rank1_agreement'] = (tp[:, 0] == tf[:, 0]).astype(float)
        sp = pq[pq['cell'] == f'{PUB}/{m}']
        sf = pq[pq['cell'] == f'{FP32}/{m}']
        for k in k_values:
            a = sp[sp['k'] == str(k)].sort_values('query_index', key=lambda s: s.astype(int))
            b = sf[sf['k'] == str(k)].sort_values('query_index', key=lambda s: s.astype(int))
            if len(a) != n or len(b) != n or not (a['query_index'].to_numpy() == b['query_index'].to_numpy()).all():
                raise CompareError(f'per_query_predictions rows for {m} k={k} do not cover the population')
            if not (a['slide_id'].to_numpy() == np.asarray(popd['ids'], dtype=str)).all():
                raise CompareError(f'per_query_predictions slide order differs from the population ({m}, k={k})')
            for rule in ('plurality', 'oracle'):
                cols[f'pred_agree_{rule}_k{k}'] = (a[f'pred_{rule}'].to_numpy() ==
                                                   b[f'pred_{rule}'].to_numpy()).astype(float)
        base = {'method': np.full(n, m), 'query_index': np.arange(n), 'slide_id': np.asarray(popd['ids'], str),
                'patient_id': np.asarray(popd['pat'], str), 'group': np.asarray(popd['lab'], str)}
        df = pd.DataFrame({**base, **{c: (v.astype(int) if c.startswith(('rank1', 'pred_')) else v)
                                      for c, v in cols.items()}})
        per.append(df)
        for c, v in cols.items():
            summ.append({'method': m, 'quantity': c, 'n': n, 'mean': _r(np.mean(v)), 'median': _r(np.median(v)),
                         'p5': _r(np.percentile(v, 5)), 'fraction_eq_1': _r(np.mean(v == 1.0)),
                         'count_eq_1': int((v == 1.0).sum())})
    allq = pd.concat(per, ignore_index=True)
    raw = allq.to_csv(index=False, lineterminator='\n', float_format='%.6f').encode()
    return A.gz_compress_like_rr5(raw), summ


# ----------------------------------------------------------------------------------------------
# A3 independent cross-cell check (Run B present)
# ----------------------------------------------------------------------------------------------

class _XlRowsStub:
    """winners_and_panels reads xl['rows'][(cid, k)] Macro F1 of the C32 cells only for the submitted-count rows
    (row_type 'count', tie_order 'submitted_xlsx'); those rows are not compared by A3 and are discarded here."""

    def __getitem__(self, key):
        return {'Overall Metrics': {'Macro F1': 0.0}}


def _xl_stub():
    return {'rows': _XlRowsStub()}


def _stat_from_row(ns, r):
    s = ns.Stat.__new__(ns.Stat)
    s.point = float(r['value'])
    s.lo = float(r['ci_low']) if r['ci_low'] != '' else np.nan
    s.hi = float(r['ci_high']) if r['ci_high'] != '' else np.nan
    s.status = r['ci_status']
    s.reps = np.full(1, np.nan)
    s.n_zero_pred = 0
    s.n_zero_support = 0
    return s


def _map_id(v):
    return FP32 + v[len(PUB):] if v.startswith(f'{PUB}/') else v


def a3_check(ns, run_a, rb_dir, run001, strict=False):
    rb = json.load(open(os.path.join(rb_dir, 'rb_spec.json')))['full_spec']
    cells = [c for c in rb['cells'] if not c['cell_id'].startswith(f'{PUB}/')]
    n_pub = len(rb['cells']) - len(cells)
    n_fp = sum(c['cell_id'].startswith(f'{FP32}/') for c in cells)
    if n_pub != 4 or n_fp != 4:
        raise CompareError(f'A3: rb_spec has {n_pub} virchow2/ and {n_fp} virchow2_fp32/ cells, expected 4 + 4')
    cells_by_id = {c['cell_id']: c for c in cells}
    _, t3 = records(A.read_product(run_a, 'table_s3_revised.csv'))
    idx = {(r['cell_id'], r['k'], r['rule'], r['tie_order']): r for r in t3 if r['metric'] == 'macro_f1'}
    slim = {}
    for cid in cells_by_id:
        slim[cid] = {}
        for order in ns.ORDERS:
            for k in ns.K_VALUES:
                for rule in ns.RULES:
                    r = idx.get((cid, str(k), rule, order))
                    if r is None:
                        raise CompareError(f'A3: no Run A table_s3 macro_f1 row for {cid} k={k} {rule} {order}')
                    slim[cid][(order, k, rule, 'macro_f1')] = _stat_from_row(ns, r)
    win, _panel, _agg, _cc = ns.winners_and_panels(rb, cells_by_id, slim, _xl_stub())
    f2b = ns.fig2b_rows(rb, cells_by_id, slim)
    rec_w = records(ns._csv_bytes(pd.DataFrame(win).astype({'k': 'Int64', 'n_tied_winners': 'Int64'})))
    rec_b = records(ns._csv_bytes(pd.DataFrame(f2b)))
    fin_w = records(A.read_product(rb_dir, 'winners.csv'))
    fin_b = records(A.read_product(rb_dir, 'fig2b_matrix.csv'))
    fails, rep = [], {}

    def sel_w(rows):
        return {(r['k'], r['rule'], r['tie_order'], r['winner_cell']): r for r in rows
                if r['row_type'] == 'slot' and r['slot_set'] == 'C32' and r['winner_cell'].startswith(f'{FP32}/')}

    def sel_c(rows):
        # F6: count rows derived by RR-5 from the slot winners (weights 1/n_tied); xlsx-sourced rows excluded
        out = {}
        for r in rows:
            if r['row_type'] == 'count' and r['tie_order'] != A3_COUNT_SKIP_ORDER:
                key = (r['slot_set'], r['rule'], r['tie_order'], r['winner'])
                if key in out:
                    raise CompareError(f'A3: duplicate count row {key}')
                out[key] = r
        return out

    def sel_b(rows, relabel):
        out = {}
        for r in rows:
            if not r['row_type'].startswith('range_'):
                continue
            r = dict(r)
            if relabel and r['scope'] in A.A8_SCOPE_MAP:
                r['scope'] = A.A8_SCOPE_MAP[r['scope']]
            out[(r['row_type'], r['display_encoder'], r['aggregation'], r['rule'], r['tie_order'], r['scope'])] = r
        return out

    for name, label, (hr, rr), (hfin, rfin), sel in (
            ('winners.csv', 'winners.csv', rec_w, fin_w, lambda rows, final: sel_w(rows)),
            ('winners.csv', 'winners.csv:count', rec_w, fin_w, lambda rows, final: sel_c(rows)),
            ('fig2b_matrix.csv', 'fig2b_matrix.csv', rec_b, fin_b, lambda rows, final: sel_b(rows, not final))):
        if hr != hfin:
            fails.append(f'A3: {name}: recomputed header != Run B header')
        a, b = sel(rr, False), sel(rfin, True)
        derived = () if (strict or label.endswith(':count')) else A3_DERIVED[name]
        n_fields, n_derived_ulp = 0, 0
        if set(a) != set(b):
            lab = 'six_dp_tie' if any(int(a[x]['n_tied_winners'] or 1) > 1 for x in a if 'n_tied_winners' in a[x]) \
                else 'identity_differs'
            fails.append(f'A3: {label}: recomputed rows {sorted(set(a) - set(b))[:4]} vs Run B rows '
                         f'{sorted(set(b) - set(a))[:4]} ({lab})')
        for key in sorted(set(a) & set(b)):
            for col in hr:
                n_fields += 1
                x, y = a[key][col], b[key][col]
                if x == y:
                    continue
                if col in derived and x != '' and y != '' and abs(float(x) - float(y)) <= A3_TOL:
                    n_derived_ulp += 1
                    continue
                fails.append(f'A3: {label} {key} {col}: recomputed {x!r} != Run B {y!r}')
        rep[label] = {'rows_recomputed': len(a), 'rows_run_b': len(b), 'fields_compared': n_fields,
                     'derived_fields_within_one_6dp_unit': n_derived_ulp, 'derived_columns_tolerated': list(derived)}
    # virchow2_fp32 argmax_k rows: per-cell, unrounded in both runs -> Run B line == Run A line
    ak, fk = argmax_k_identity(run_a, rb_dir)
    rep['argmax_k_fp32_vs_run_a'] = ak
    fails += fk
    nf, fn = non_fp32_identity(run001, rb_dir)
    rep['non_fp32_identity'] = nf
    fails += fn
    rep.update({'cells_used': len(cells), 'virchow2_fp32_cells': n_fp, 'non_virchow2_cells': len(cells) - n_fp,
                'source': 'Run A table_s3_revised.csv at RR-5 6-dp writer values', 'strict': strict,
                'tolerance': A3_TOLERANCE_NOTE, 'six_dp_tie': SIX_DP_TIE_NOTE,
                'pass': not fails, 'failures': fails[:200]})
    return rep, fails


def _row_lines(data):
    head, lines = A.split_lines(data)
    hf = A.parse_fields(head)
    return head, hf, [(ln, dict(zip(hf, A.parse_fields(ln)))) for ln in lines]


def argmax_k_identity(run_a, rb_dir):
    out, fails = {}, []
    _, _, la = _row_lines(A.read_product(run_a, 'winners.csv'))
    _, _, lb = _row_lines(A.read_product(rb_dir, 'winners.csv'))
    sa = [ln for ln, r in la if r['row_type'] == 'argmax_k' and r['winner_cell'].startswith(f'{FP32}/')]
    sb = [ln for ln, r in lb if r['row_type'] == 'argmax_k' and r['winner_cell'].startswith(f'{FP32}/')]
    ok = bool(sa) and sa == sb
    if not ok:
        fails.append(f'A3: winners.csv virchow2_fp32 argmax_k rows: Run B ({len(sb)}) != Run A ({len(sa)})')
    out = {'rows': len(sa), 'identical': ok}
    return out, fails


def non_fp32_identity(run001, rb_dir):
    """F6 (A4-style): Run B winners.csv / fig2b_matrix.csv with every virchow2_fp32/ line removed == run_001 with
    every virchow2/ line removed, byte for byte and in order, excluding on both sides only the F6_DEPENDENT rows
    (which aggregate across encoders, carry the A8 scope relabel, and are recomputed by A3 above)."""
    out, fails = {}, []
    for name, dep in F6_DEPENDENT.items():
        h0, _, l0 = _row_lines(A.read_product(run001, name))
        h1, _, l1 = _row_lines(A.read_product(rb_dir, name))
        k0 = [ln for ln, r in l0 if f'{PUB}/'.encode() not in ln and not dep(r)]
        k1 = [ln for ln, r in l1 if f'{FP32}/'.encode() not in ln and not dep(r)]
        ok = h0 == h1 and k0 == k1
        out[name] = {'identical': ok, 'run001_rows_compared': len(k0), 'run_b_rows_compared': len(k1),
                     'run001_virchow2_rows_removed': sum(f'{PUB}/'.encode() in ln for ln, _ in l0),
                     'run_b_virchow2_fp32_rows_removed': sum(f'{FP32}/'.encode() in ln for ln, _ in l1),
                     'excluded_dependent_rows_run001': sum(1 for ln, r in l0 if f'{PUB}/'.encode() not in ln
                                                           and dep(r)),
                     'excluded_dependent_rows_run_b': sum(1 for ln, r in l1 if f'{FP32}/'.encode() not in ln
                                                          and dep(r))}
        if not ok:
            first = next((i for i, (x, y) in enumerate(zip(k0, k1)) if x != y), min(len(k0), len(k1)))
            fails.append(f'F6: {name}: Run B rows without virchow2_fp32/ != run_001 rows without virchow2/ '
                         f'(first difference at retained row {first}; {len(k1)} vs {len(k0)} rows)')
    return out, fails


# ----------------------------------------------------------------------------------------------
# product_changes.csv (Major-6)
# ----------------------------------------------------------------------------------------------

def _changes(product, ctype, before, after, value_of=None):
    out = []
    for key in sorted(set(before) | set(after), key=lambda t: tuple(map(str, t))):
        b, a = before.get(key), after.get(key)
        if b == a:
            continue
        bv = av = ''
        if value_of:
            bv, av = value_of[0].get(key, ''), value_of[1].get(key, '')
        out.append({'product': product, 'change_type': ctype, 'key': '|'.join(map(str, key)),
                    'before': '' if b is None else b, 'after': '' if a is None else a,
                    'before_value': bv, 'after_value': av})
    return out


def product_changes(run001, rb_dir):
    out = []
    _, w0 = records(A.read_product(run001, 'winners.csv'))
    _, w1 = records(A.read_product(rb_dir, 'winners.csv'))

    def slot(rows, mp):
        d, v, ru, ov = {}, {}, {}, {}
        for r in rows:
            if r['row_type'] != 'slot':
                continue
            key = (r['slot_set'], r['display_encoder'], r['k'], r['rule'], r['tie_order'])
            d.setdefault(key, []).append(mp(r['winner_cell']))
            v[key] = r['value']
            ru[key] = r['runner_up']
            ov.setdefault(key, []).append(r['winner_ci_overlaps_runner_up'])
        return ({k: ';'.join(sorted(x)) for k, x in d.items()}, v, ru, {k: ';'.join(x) for k, x in ov.items()})

    s0, v0, r0, o0 = slot(w0, _map_id)
    s1, v1, r1, o1 = slot(w1, lambda x: x)
    out += _changes('winners.csv', 'slot_winner', s0, s1, (v0, v1))
    out += _changes('winners.csv', 'slot_runner_up', r0, r1, (v0, v1))
    out += _changes('winners.csv', 'slot_winner_ci_overlaps_runner_up', o0, o1, (v0, v1))

    def counts(rows):
        return {(r['slot_set'], r['rule'], r['tie_order'], r['winner']): r['weight'] for r in rows
                if r['row_type'] == 'count'}
    out += _changes('winners.csv', 'count_weight', counts(w0), counts(w1))

    def argk(rows, mp):
        return {(mp(r['winner_cell']), r['rule'], r['tie_order']): r['winner'] for r in rows
                if r['row_type'] == 'argmax_k'}
    out += _changes('winners.csv', 'argmax_k', argk(w0, _map_id), argk(w1, lambda x: x))

    _, p0 = records(A.read_product(run001, 'fig2a_panel.csv'))
    _, p1 = records(A.read_product(rb_dir, 'fig2a_panel.csv'))

    def panel(rows, mp, col):
        return {(r['slot_set'], r['display_encoder'], r['k'], r['rule'], r['tie_order'], mp(r['cell_id'])): r[col]
                for r in rows}
    vals = (panel(p0, _map_id, 'value'), panel(p1, lambda x: x, 'value'))
    out += _changes('fig2a_panel.csv', 'argmax_is_winner', panel(p0, _map_id, 'is_winner'),
                    panel(p1, lambda x: x, 'is_winner'), vals)
    out += _changes('fig2a_panel.csv', 'rank', panel(p0, _map_id, 'rank'), panel(p1, lambda x: x, 'rank'), vals)

    _, b0 = records(A.read_product(run001, 'fig2b_matrix.csv'))
    _, b1 = records(A.read_product(rb_dir, 'fig2b_matrix.csv'))

    def rng(rows, mp, relabel, col):
        out_ = {}
        for r in rows:
            if not r['row_type'].startswith('range_'):
                continue
            scope = A.A8_SCOPE_MAP.get(r['scope'], r['scope']) if relabel else r['scope']
            key = (r['row_type'], r['display_encoder'], r['aggregation'], r['rule'], r['tie_order'], scope)
            out_[key] = mp(r[col]) if col.endswith('_cell') else r[col]
        return out_
    bv = (rng(b0, _map_id, True, 'value'), rng(b1, lambda x: x, False, 'value'))
    for col in ('argmax_cell', 'argmin_cell', 'value'):
        out += _changes('fig2b_matrix.csv', f'range_{col}', rng(b0, _map_id, True, col),
                        rng(b1, lambda x: x, False, col), bv)

    _, c0 = records(A.read_product(run001, 'fig2c_groups.csv'))
    _, c1 = records(A.read_product(rb_dir, 'fig2c_groups.csv'))

    def f2c(rows, mp):
        return {(mp(r['pipeline']), r['group']): r['revised_rank'] for r in rows}

    def f2cv(rows, mp):
        return {(mp(r['pipeline']), r['group']): r['revised_order_mean_plurality_f1'] for r in rows}
    out += _changes('fig2c_groups.csv', 'revised_rank', f2c(c0, _map_id), f2c(c1, lambda x: x),
                    (f2cv(c0, _map_id), f2cv(c1, lambda x: x)))
    return out


PENDING_CHANGE = {'product': 'all cross-cell replacement products', 'change_type': 'pending_run_b',
                  'key': '', 'before': '', 'after': '', 'before_value': '', 'after_value': ''}


# ----------------------------------------------------------------------------------------------
# M6 lineage table
# ----------------------------------------------------------------------------------------------

LINEAGE_COLS = ('method', 'aggregation', 'k', 'rule', 'metric', 'tie_order', 'submitted_value_xlsx',
                'submitted_ci_xlsx', 'published_matrix_value', 'published_matrix_ci_low', 'published_matrix_ci_high',
                'published_matrix_ci_status', 'fp32_value', 'fp32_ci_low', 'fp32_ci_high', 'fp32_ci_status',
                'delta_fp32_minus_published', 'paired_ci_low', 'paired_ci_high', 'paired_ci_status',
                'material_flag', 'flag_role', 'published_matrix_sha256', 'fp32_matrix_sha256',
                'bank_identity_published', 'bank_identity_fp32', 'nbr_median_top10_jaccard', 'nbr_rank1_agreement',
                'manuscript_location', 'rule_definition', 'submitted_ci_method', 'revised_ci_method',
                'submitted_bank', 'revised_bank', 'rr5_run_id_published', 'rr5_run_id_fp32',
                'value_used_in_revised_manuscript', 'author_decision')


def rr5_run_id_fp32(run_dir, out=None):
    """ADDENDUM_001: '<run-dir relative to the canonical output root>/rr5_additive_001'. The root is the adapter's
    OUT constant (= the independent PROSPER_OUT), not the job-time PROSPER_OUT env var (submit sets that to the step
    dir). Both sides are realpath'ed so the /PROSPER_STORAGE and /PROSPER_STORAGE mount aliases give the same string."""
    out = A.OUT if out is None else out
    rel = os.path.relpath(os.path.realpath(run_dir), os.path.realpath(out))
    return '/'.join(rel.split(os.sep) + ['rr5_additive_001'])


def lineage_table(ns, spec, run_a, pe, paired, summ, methods, run_dir):
    cells = {c['cell_id']: c for c in spec['cells']}
    _, t3 = records(A.read_product(run_a, 'table_s3_revised.csv'))
    sub = {(r['cell_id'], r['k'], r['rule'], r['metric']): r for r in t3 if r['tie_order'] == 'canonical'}
    pidx = {(r['method'], r['k'], r['rule'], r['metric']): r for r in paired}
    nb = {(s['method'], s['quantity']): s for s in summ}
    rr5_id = rr5_run_id_fp32(run_dir)
    rows = []
    for r in pe:
        if r['tie_order'] != 'canonical':
            continue
        m, k, rule, metric = r['method'], r['k'], r['rule'], r['metric']
        p = pidx[(m, k, rule, metric)]
        s = sub.get((f'{PUB}/{m}', str(k), rule, metric), {})
        sv = s.get('submitted_value', '')
        sci = f"[{s.get('submitted_ci_low', '')}, {s.get('submitted_ci_high', '')}]" if s.get('submitted_ci_low') \
            else ''
        agg = ns.AGG_DISPLAY[m]
        rows.append({'method': m, 'aggregation': agg, 'k': k, 'rule': rule, 'metric': metric,
                     'tie_order': 'canonical', 'submitted_value_xlsx': sv, 'submitted_ci_xlsx': sci,
                     'published_matrix_value': r['published_value'], 'published_matrix_ci_low': r['published_ci_low'],
                     'published_matrix_ci_high': r['published_ci_high'],
                     'published_matrix_ci_status': r['published_ci_status'], 'fp32_value': r['fp32_value'],
                     'fp32_ci_low': r['fp32_ci_low'], 'fp32_ci_high': r['fp32_ci_high'],
                     'fp32_ci_status': r['fp32_ci_status'], 'delta_fp32_minus_published': p['D_value_6dp'],
                     'paired_ci_low': p['D_ci_low_6dp'], 'paired_ci_high': p['D_ci_high_6dp'],
                     'paired_ci_status': p['D_status'], 'material_flag': p['material_flag'],
                     'flag_role': p['flag_role'] if metric == 'macro_f1' else '',
                     'published_matrix_sha256': cells[f'{PUB}/{m}']['sha256'],
                     'fp32_matrix_sha256': cells[f'{FP32}/{m}']['sha256'],
                     'nbr_median_top10_jaccard': nb[(m, 'jaccard_top10')]['median'],
                     'nbr_rank1_agreement': nb[(m, 'rank1_agreement')]['mean'],
                     'manuscript_location': f'Table S3 row Virchow2/{agg} k={k} ({rule}, {metric})',
                     'rule_definition': RULE_DEFINITION[rule], 'value_used_in_revised_manuscript': r['fp32_value'],
                     **{c: LINEAGE_CONST[c] for c in LINEAGE_COLS if c in LINEAGE_CONST}})
        rows[-1]['bank_identity_published'] = LINEAGE_CONST['bank_identity_published']
        rows[-1]['bank_identity_fp32'] = LINEAGE_CONST['bank_identity_fp32']
        rows[-1]['rr5_run_id_fp32'] = rr5_id
    return rows


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------

def do_compare(ns, run_dir, run001, n_reps=None, run_b_pending=False, a3_strict=False):
    run_a, spec, sums_a = require_run_a(run_dir)
    rb_dir = run_b_state(run_dir, run_b_pending)
    out = os.path.join(run_dir, 'comparison_001')
    A.assert_writable_dir(out)
    n_reps = int(spec['n_reps'] if n_reps is None else n_reps)
    methods, k_values, orders = tuple(ns.METHODS), tuple(ns.K_VALUES), tuple(ns.ORDERS)
    cids = [f'{e}/{m}' for e in (FP32, PUB, 'uni2') for m in methods]
    keep = {}
    stats, ctx = A.recompute_cell_stats(ns, spec, cids, os.path.join(run_dir, 'comparison_001_recompute_bindings'),
                                        n_reps, keep=keep)
    contract_draws = ctx['draws_sha256'] == A.DRAWS_SHA and ctx['n_reps'] == 5000
    mode = BOOTSTRAP_MODE if contract_draws else (f'whole-patient multiplicity weights, n_reps = {ctx["n_reps"]}, '
                                                  f'draws sha256 {ctx["draws_sha256"]} (NOT draws #46)')
    report = {'contract': 'contract_candidate_002 + LEAD_RESOLUTIONS_002(_ADDENDUM)', 'bootstrap_mode': mode,
              'n_reps': ctx['n_reps'], 'draws_sha256': ctx['draws_sha256'], 'contract_draws': contract_draws,
              'cells_recomputed': cids, 'one_draws_load_one_population_ordering': True,
              'W_sha256': ctx['W_sha256'], 'W_shape': ctx['W_shape'], 'W_dtype': ctx['W_dtype'],
              'W_identity': ('one W object passed to RR-5 cell_statistics for all recomputed cells (F5); A5 detects '
                             'patient-order, not replicate-order, misalignment'),
              'run_a_sha256sums': sums_a,
              'run_b': 'pending' if rb_dir is None else 'present'}
    if rb_dir is not None:
        fr = os.path.join(rb_dir, 'faithfulness_report.json')
        report['run_b_faithfulness_report'] = {'path': fr, 'sha256': A.sha256_file(fr)}
    fails = []
    a5, f = a5_check(ns, stats, run_a, k_values, orders)
    report['a5'] = a5
    fails += f
    ri, f = routine_identity(ns, stats, run001, contract_draws)
    report['routine_identity'] = ri
    fails += f
    if rb_dir is not None:
        a3, f = a3_check(ns, run_a, rb_dir, run001, strict=a3_strict)
        fails += f
    else:
        a3 = {'status': 'pending', 'note': 'Run B infeasible or not released; cross-cell replacement products are '
                                           'pending and are not built from Run A (A3)', 'six_dp_tie': SIX_DP_TIE_NOTE}
    report['a3'] = a3
    if fails:
        report['pass'] = False
        report['failures'] = fails[:200]
        os.makedirs(out, exist_ok=True)
        A.write_new(os.path.join(out, 'compare_report.json'), A.canonical_json(report))
        raise CompareError(f'comparison gates failed: {fails[:5]}')
    pe = point_estimates(ns, stats, methods, k_values, orders)
    paired, mat = paired_differences(ns, stats, methods, k_values, mode)
    mj = materiality(mat, paired, mode, ctx['n_reps'])
    pc = per_class(ns, stats, methods, k_values, ctx['labels_order'])
    gz, summ = neighbour_agreement(keep, ctx['popd'], run_a, methods, k_values)
    changes = product_changes(run001, rb_dir) if rb_dir is not None else [PENDING_CHANGE]
    lin = lineage_table(ns, spec, run_a, pe, paired, summ, methods, run_dir)
    files = {
        'point_estimates.csv': _csv(pe, pe[0].keys()),
        'paired_differences.csv': _csv(paired, PAIRED_COLS),
        'neighbour_agreement_per_query.csv.gz': gz,
        'neighbour_agreement_summary.csv': _csv(summ, summ[0].keys()),
        'per_class_comparison.csv': _csv(pc, pc[0].keys()),
        'materiality.json': A.canonical_json(mj),
        'product_changes.csv': _csv(changes, PRODUCT_CHANGE_COLS),
        'lineage_table_S3_virchow2.csv': _csv(lin, LINEAGE_COLS),
        'lineage_table_S3_virchow2.footnotes.md': LINEAGE_FOOTNOTES.encode(),
    }
    os.makedirs(out, exist_ok=True)
    for name, data in files.items():
        A.write_new(os.path.join(out, name), data)
    report.update({'pass': True, 'package_flag': mj['package_flag'], 'n_rows': {
        'point_estimates': len(pe), 'paired_differences': len(paired), 'per_class_comparison': len(pc),
        'product_changes': len(changes), 'lineage': len(lin)},
        'outputs': {n: A.sha256_bytes(d) for n, d in files.items()}})
    A.write_new(os.path.join(out, 'compare_report.json'), A.canonical_json(report))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rr5', default=A.RR5_PATH)
    ap.add_argument('--rr5-sha', default=A.RR5_SHA)
    ap.add_argument('--run-dir', default=os.path.join(A.ROOT, A.RUN_STAGE))
    ap.add_argument('--run001', default=A.RUN001)
    ap.add_argument('--n-reps', type=int, default=None, help='default: the Run A spec n_reps (5000)')
    ap.add_argument('--run-b-pending', action='store_true',
                    help='Run B infeasible/not released: report cross-cell replacement products as pending (A3)')
    ap.add_argument('--a3-strict', action='store_true',
                    help='REFUSED: tests only (function argument a3_strict); the production A3 tolerance is fixed '
                         'by LEAD_IMPL_DECISIONS_001')
    args = ap.parse_args(argv)
    try:
        if args.a3_strict:
            raise CompareError('--a3-strict is refused outside tests (LEAD_IMPL_DECISIONS_001 A3 tolerance)')
        A.require_pinned_rr5(args.rr5, args.rr5_sha)
        ns = A.load_rr5(args.rr5, args.rr5_sha)
        A.add_fp32_display(ns)
        rep = do_compare(ns, os.path.abspath(args.run_dir), args.run001, args.n_reps, args.run_b_pending)
    except CompareError as e:
        print(f'STOP: {e}', file=sys.stderr)
        return 1
    print(json.dumps({'pass': rep['pass'], 'package_flag': rep['package_flag']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
