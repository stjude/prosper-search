#!/usr/bin/env python3
"""R2.1 B(c): fair-design top-5 vote F1 relative error reduction (RER) on the common population.

Implements [decision record] "### B(c) - Fair-design top-5 F1 relative error reduction (RER) ... candidate 002"
(items 1-13, outputs, gates and pinned hashes). Descriptive only; no equivalence claim.

Population: 2,604 slides / 1,800 patients / 14 groups. Twelve contrasts (filtered arm vs all-tissue
arm of the same encoder and method): route (a) legacy_tumour_only for UNI2 and Virchow2 (OT-cosine,
Yottixel); H-Optimus query_only / both_filtered at budgets 5.0 and 100.0 (OT-cosine, Yottixel).

Rules: plurality (primary, non-oracle) and legacy_generous (ORACLE, reconciliation / like-for-like
with Table S5 only), both imported from the frozen scripts/r2_3_internal_failure_tables.py bytes.
Non-returned queries: "miss" (primary; prediction NONE -> FN for the truth class only) and
"both_returned" (secondary; the query is dropped from both arms of that contrast).
Uncertainty: whole-patient paired bootstrap on the saved patient_draws.npy (5000 x 1800); each
query is re-weighted by its patient's draw count; 95 % linear percentile intervals.

Exit codes: 0 = all gates pass (COMPLETE.json); 1 = a gate failed (FAILED.json); 2 = usage error.
"""
import argparse
import datetime
import gzip
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import platform
import socket
import sys
import traceback

import numpy as np
import pandas as pd

K = 5
N_REPS = 5000
N_PATIENTS = 1800
N_QUERIES = 2604
N_GROUPS = 14
UNDEFINED_THRESHOLD = 0.005  # 1 - F1_all below this -> RER undefined (fixed, item 3)
SUBMITTED_EPS = 1e-8  # make_figure_3b.py compute_effects error_reduction guard
RECON_DELTA_TOL = 1e-6
RECON_F1_TOL = 1e-9
QUANTILES = (0.025, 0.975)
QUANTILE_METHOD = 'linear'
NONE_LABEL = 'NONE'
RULES = ('plurality', 'legacy_generous_ORACLE')
HANDLINGS = ('miss', 'both_returned')
METHODS = ('ot_cosine', 'yottixel')
ALLOWED_STATUS = ('returned', 'empty_query')
PRESPECIFIED_GROUPS = ('LIVER', 'ACT/ACC', 'MEL', 'NF', 'MRT', 'MPNST', 'DSRCT')  # item 8(d)
MACRO = 'MACRO'

PIN = {
    'population': 'c7cae1f0d855952e84e85aa90c7ef7d8a622086273c113285655c96be946d97b',
    'ranked_lists:uni2': 'f908c7efe6d52b9ff2b6ebcfe030daba36bcaf9d49d514ce7b60702fc52fd756',
    'ranked_lists:virchow2': 'cfefb6c08e4d700c17954f36503504ad94fae36311f00139f4e06da2051a98cc',
    'ranked_lists:hoptimus': 'd85aa06be07447b210c87241af6cb27c5b3be319913e3135f3860b1c4e495618',
    'query_metrics:uni2': 'f7dc994f8dd782d20f98770b36f86a8e1743edb928e960990fc10e480e6d8872',
    'query_metrics:virchow2': 'c7088ebeae3a6c5430172f812b6de13cd8d7f7b157f66ae5e508f088e3d30b4b',
    'query_metrics:hoptimus': '885be2a3a41c1ffb5119ce7e7618ef14ea5ac833274ff620a292fc9794219f04',
    'patient_draws:uni2': 'cfaed70d0c11084fb72b4dd7c787313b5da9d6e47f374803d2671b647886f350',
    'patient_draws:virchow2': 'cfaed70d0c11084fb72b4dd7c787313b5da9d6e47f374803d2671b647886f350',
    'patient_draws:hoptimus': 'cfaed70d0c11084fb72b4dd7c787313b5da9d6e47f374803d2671b647886f350',
    # item 12 / "Pinned historical hashes": pinned by the analyst from the frozen packages
    'summary:uni2': '9b655ec0855a68c09aaa93873d31656e906943865622964ec471247db3f32559',
    'summary:virchow2': '01bcb305d16d07994ff662456e673168386a87bf0636cb5c82156bf140eb44d2',
    'summary:hoptimus': 'b6abd7c9a335aa04dac38bad2a555213c20ef1a66c434515afd7c71934f49545',
    'hist:uni2:ot_cosine:full': 'a4c135d4f5d8a1189327688cfbe0206370c026b149b6d123cf8dc94b921439ef',
    'hist:uni2:ot_cosine:tumour': '2c884e3c9f6f0775a4df700469188519333211929ca45922fe2c077551e9ba78',
    'hist:uni2:yottixel:full': '1303ccd5325e6ad4445124546e47b20e0a6306606a5b84af32a779423b5843c7',
    'hist:uni2:yottixel:tumour': '3517e6abfd46164824238e5f725a4bf02cfb1b7134086a21275ca0369a4ac62f',
    'hist:virchow2:ot_cosine:full': '3ab6867fcc87ebb53e3901bb8437dcab4d34219fd3ba3f526e10896dcf6dd14d',
    'hist:virchow2:ot_cosine:tumour': '2e0e9a1946abc7ac8e329b4ed7c515ff8aa14b4ab3a15f2e9a24cdcd2b24a0c5',
    'hist:virchow2:yottixel:full': '5d5ac9e6672538e0e23af82e5150c9dde5c5a2bf6eeaa942d3b449818697d822',
    'hist:virchow2:yottixel:tumour': '47bf1b99d78fc4b7bce28b19ad4f17ead7f96ae758347cfb4cc98580d2b21dfd',
    'table_s5': '86b341f5cf5e30fe0bed4e25328a4a88b22f5eb11206a992951732c45ce3cf7b',
    'vote_module': 'afb1c4bd3220ece353268933203b80bb1cd9d5554bd88de9083ab0c27e126eb0',
}

PACKAGE_DIRS = {
    'uni2': ('r2_1_legacy_filter_20261011', 'eval_uni2', 'run_001'),
    'virchow2': ('r2_1_legacy_filter_20261011', 'eval_virchow2', 'run_001'),
    'hoptimus': ('r2_1_filter_20261003', 'evaluation_qf'),
}

# Raw stored budget strings per package (CSV text; NaN = empty) -> canonical label (item 13).
RAW_BUDGETS = {'uni2': {'legacy'}, 'virchow2': {'legacy'}, 'hoptimus': {'5', '100'}}
BUDGET_CANON = {'legacy': 'legacy', '5': '5.0', '5.0': '5.0', '100': '100.0', '100.0': '100.0'}

# Expected arms per package: (arm, canonical budget) ; '' = all_tissue (stored NaN)
ARMS = {
    'uni2': [('all_tissue', ''), ('legacy_tumour_only', 'legacy')],
    'virchow2': [('all_tissue', ''), ('legacy_tumour_only', 'legacy')],
    'hoptimus': [('all_tissue', ''), ('query_only', '5.0'), ('query_only', '100.0'),
                 ('both_filtered', '5.0'), ('both_filtered', '100.0')],
}

# Item 2 / BC-4: exact non-return sets (identical for both methods).
EXPECTED_NONRETURN = {
    ('uni2', 'all_tissue', ''): set(),
    ('uni2', 'legacy_tumour_only', 'legacy'): {16738},
    ('virchow2', 'all_tissue', ''): set(),
    ('virchow2', 'legacy_tumour_only', 'legacy'): {14601, 15678},
    ('hoptimus', 'all_tissue', ''): set(),
    ('hoptimus', 'query_only', '5.0'): {14601, 15678, 15844, 16738},
    ('hoptimus', 'both_filtered', '5.0'): {14601, 15678, 15844, 16738},
    ('hoptimus', 'query_only', '100.0'): {14601, 15678, 15794, 16737, 16738},
    ('hoptimus', 'both_filtered', '100.0'): {14601, 15678, 15794, 16737, 16738},
}
EXPECTED_NONRETURN_TRUTH = {16738: 'RB', 14601: 'LIVER', 15678: 'OS'}

# Item 9: pre-registered OT lineage expectation, historical minus reproduced (truth->truth cells).
LINEAGE_OT_EXPECTATION = {'uni2': {('RB', 'RB'): 1}, 'virchow2': {('LIVER', 'LIVER'): 1, ('OS', 'OS'): 1}}

ENC_S5 = {'uni2': 'UNI2', 'virchow2': 'Virchow2'}
METHOD_S5 = {'ot_cosine': 'OT-cos', 'yottixel': 'Yottixel'}
SUBMITTED_NOTE = ('submitted Table S5: generous (oracle) rule; full-slide arm 2,778 slides / 15 groups incl. '
                  'NORMAL queries and references vs tumour-only 2,604 / 14; slide-level analysis')
DISCLOSURE_LINE = ('the submitted tumour-only analysis scored the 1 (UNI2) / 2 (Virchow2) slides with no retained '
                   'tumour tiles as correct via same-patient library slides; the fair design scores them as misses '
                   '(primary) or drops them (secondary).')

KEY_COLS = ['encoder', 'method', 'arm', 'design', 'budget', 'rule', 'handling', 'contrast_id']


class GateError(Exception):
    """A specification gate failed; the job writes FAILED.json and exits 1."""

    def __init__(self, gate, reason, detail=None):
        super().__init__(f'{gate}: {reason}')
        self.gate, self.reason, self.detail = gate, reason, detail


# --------------------------------------------------------------------------------------------
# Hashing and the frozen vote module (item 13)
# --------------------------------------------------------------------------------------------

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    size = 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


class _BytesLoader(importlib.abc.Loader):
    def __init__(self, source, origin):
        self.source, self.origin = source, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # The frozen module derives its repository root from __file__ at import time.
        module.__dict__.setdefault('__file__', self.origin)
        exec(compile(self.source, self.origin, 'exec'), module.__dict__)


def load_vote_functions(path, expected_sha=PIN['vote_module'], source=None):
    """Read the bytes once, verify the full sha256, exec those same bytes; return (plurality, legacy_generous, info).

    If `source` is given it must be the bytes already read from `path`; the path is then not opened again.
    """
    if source is None:
        with open(path, 'rb') as f:
            source = f.read()
    got = sha256_bytes(source)
    if got != expected_sha:
        raise GateError('vote_module_hash', f'{path} sha256 {got} != pinned {expected_sha}')
    loader = _BytesLoader(source, str(path))
    spec = importlib.util.spec_from_loader('r2_3_internal_failure_tables_frozen', loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.plurality, module.legacy_generous, {'path': str(path), 'sha256': got, 'size': len(source)}


def vote(labels, truth, rule, plurality, legacy_generous):
    """Top-5 vote of a non-empty label list (ranks 1..5)."""
    top = list(labels[:K])
    if len(top) != K:
        raise GateError('vote_input', f'expected {K} labels, got {len(top)}')
    if rule == 'plurality':
        return plurality(top, K)
    if rule == 'legacy_generous_ORACLE':
        return legacy_generous(top, truth, K)[1]
    raise ValueError(rule)


# --------------------------------------------------------------------------------------------
# F1 / macro / RER arithmetic (items 3, 4, 13)
# --------------------------------------------------------------------------------------------

def f1_from_counts(tp, fp, fn, support):
    """F1 = 2TP/(2TP+FP+FN); 0 when TP = 0 with positive denominator; NaN where support == 0."""
    tp, fp, fn, support = (np.asarray(x, float) for x in (tp, fp, fn, support))
    den = 2 * tp + fp + fn
    with np.errstate(invalid='ignore', divide='ignore'):
        f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1.0), 0.0)
    return np.where(support > 0, f1, np.nan)


def f1_from_confusion(cm):
    """Per-class F1 from a square confusion matrix (rows truth, cols prediction)."""
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1.0), 0.0)


def weighted_f1(W, T, P, inc):
    """Weighted per-class F1.

    W: (R, n) query weights; T: (n, C) truth one-hot; P: (n, C) prediction one-hot (zero row = NONE);
    inc: (n,) 0/1 inclusion. Returns (f1 (R, C) NaN where weighted support is 0, support (R, C)).
    """
    W = np.atleast_2d(np.asarray(W, float))
    inc = np.asarray(inc, float)[:, None]
    Ti = T * inc
    Pi = P * inc
    support = W @ Ti
    tp = W @ (Ti * Pi)
    pp = W @ Pi
    return f1_from_counts(tp, pp - tp, support - tp, support), support


def macro_mean(f1):
    """Unweighted mean over classes with defined F1 (positive weighted support); returns (macro, n_excluded)."""
    f1 = np.asarray(f1, float)
    ok = np.isfinite(f1)
    n = ok.sum(-1)
    s = np.where(ok, f1, 0.0).sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
    return m, f1.shape[-1] - n


def rer_values(f_filt, f_all, threshold=UNDEFINED_THRESHOLD):
    """RER = (F1_filt - F1_all)/(1 - F1_all); NaN and flagged undefined where 1 - F1_all < threshold.

    Returns (rer, undefined) with undefined True only where F1_all is finite and below the threshold gap.
    """
    f_filt = np.asarray(f_filt, float)
    f_all = np.asarray(f_all, float)
    gap = 1.0 - f_all
    finite = np.isfinite(f_all) & np.isfinite(f_filt)
    undefined = finite & (gap < threshold)
    ok = finite & ~undefined
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(ok, (f_filt - f_all) / np.where(ok, gap, 1.0), np.nan)
    return r, undefined


def percentile_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    lo, hi = np.quantile(x, QUANTILES, method=QUANTILE_METHOD)
    return float(lo), float(hi)


def max_undefined_reps(n_reps):
    """Reportability limit of item 3 / BC-2: <= 1 % of replicates (50 of 5000)."""
    return n_reps // 100


def rer_interval(point_rer_undefined, rep_rer, rep_undefined, rep_zero_support, is_macro):
    """Apply the item-3 reportability rule.

    Returns dict(rer_ci_low, rer_ci_high, rer_ci_status, n_undefined_reps). Per-class intervals are reported
    only if (threshold-undefined + zero-support) replicates <= 1 %; the macro interval is always reported
    (with its undefined count) unless the point estimate is undefined.
    """
    n_reps = len(rep_rer)
    n_und = int(np.sum(rep_undefined))
    n_zero = int(np.sum(rep_zero_support))
    if point_rer_undefined:
        return dict(rer_ci_low=np.nan, rer_ci_high=np.nan, rer_ci_status='undefined_point', n_undefined_reps=n_und)
    if not is_macro and (n_und + n_zero) > max_undefined_reps(n_reps):
        return dict(rer_ci_low=np.nan, rer_ci_high=np.nan, rer_ci_status='not_estimable_undefined_gt_1pct',
                    n_undefined_reps=n_und)
    lo, hi = percentile_ci(rep_rer)
    return dict(rer_ci_low=lo, rer_ci_high=hi, rer_ci_status='reported', n_undefined_reps=n_und)


def submitted_error_reduction(f1_full, f1_tumour, eps=SUBMITTED_EPS):
    """make_figure_3b.py compute_effects(metric='error_reduction') for one cell (NaN if undefined)."""
    if not (np.isfinite(f1_full) and np.isfinite(f1_tumour)):
        return np.nan
    den = 1.0 - f1_full
    if abs(den) < eps:
        return np.nan
    return (f1_tumour - f1_full) / den


# --------------------------------------------------------------------------------------------
# Handling of non-returned queries (item 2) and tie reversal (item 10)
# --------------------------------------------------------------------------------------------

def inclusion_mask(handling, nonret_all, nonret_filt):
    """(n,) 0/1 inclusion for a contrast. nonret_*: boolean arrays."""
    nonret_all = np.asarray(nonret_all, bool)
    nonret_filt = np.asarray(nonret_filt, bool)
    if handling == 'miss':
        return np.ones(nonret_all.shape, float)
    if handling == 'both_returned':
        return (~(nonret_all | nonret_filt)).astype(float)
    raise ValueError(handling)


def prediction_matrix(preds, groups):
    """One-hot (n, C) from a prediction list; NONE -> zero row (false negative only)."""
    gi = {g: i for i, g in enumerate(groups)}
    P = np.zeros((len(preds), len(groups)))
    for i, p in enumerate(preds):
        if p == NONE_LABEL:
            continue
        if p not in gi:
            raise GateError('prediction_label', f'prediction {p!r} not in the population groups')
        P[i, gi[p]] = 1.0
    return P


def has_boundary_tie(distances, k=K):
    d = np.asarray(distances, float)
    return bool(d.size > k and d[k - 1] == d[k])


def reverse_boundary_block(labels, distances, k=K):
    """Reverse the order within the equal-distance block containing rank k when it straddles k (d_k == d_{k+1}).

    Returns (labels, has_tie). Raises GateError if the tied block reaches the last stored rank (members beyond the
    stored depth unknown) or is not contiguous.
    """
    d = np.asarray(distances, float)
    labels = list(labels)
    if not has_boundary_tie(d, k):
        return labels, False
    tied = np.flatnonzero(d == d[k - 1])
    lo, hi = int(tied[0]), int(tied[-1])
    if not np.array_equal(tied, np.arange(lo, hi + 1)):
        raise GateError('tie_block_containment', 'tied distance block is not contiguous')
    if hi >= d.size - 1:
        raise GateError('tie_block_containment', f'k={k} boundary tie block reaches the stored depth {d.size}')
    labels[lo:hi + 1] = labels[lo:hi + 1][::-1]
    return labels, True


# --------------------------------------------------------------------------------------------
# Binding checks (items 2, 12)
# --------------------------------------------------------------------------------------------

def check_draws(draws, patients_sorted, summaries, n_reps=N_REPS, n_patients=N_PATIENTS):
    """Item 12 / BC-5: column order = sorted unique population patient_id = each summary's patient_draws_patients."""
    if draws.shape != (n_reps, n_patients):
        raise GateError('draws_binding', f'shape {draws.shape} != {(n_reps, n_patients)}')
    if draws.dtype != np.int64:
        raise GateError('draws_binding', f'dtype {draws.dtype} != int64')
    if not np.all(draws.sum(1) == n_patients):
        raise GateError('draws_binding', 'row sums differ from the number of patients')
    patients_sorted = list(patients_sorted)
    if patients_sorted != sorted(set(patients_sorted)) or len(patients_sorted) != n_patients:
        raise GateError('draws_binding', 'population patient list is not sorted-unique of the expected size')
    for name, s in summaries.items():
        if list(s.get('patient_draws_patients', [])) != patients_sorted:
            raise GateError('draws_binding', f'{name}: summary.json patient_draws_patients != sorted unique '
                                             'population patient_id')
    return True


def check_nonreturn_sets(observed, expected):
    """observed/expected: dict key -> set of query ids; any difference fails."""
    bad = {}
    for key in sorted(set(observed) | set(expected), key=str):
        o, e = observed.get(key), expected.get(key)
        if o is None or e is None or set(o) != set(e):
            bad[str(key)] = {'observed': sorted(o) if o is not None else None,
                             'expected': sorted(e) if e is not None else None}
    if bad:
        raise GateError('nonreturn_sets', 'non-return ID sets differ from the contract', bad)
    return True


# --------------------------------------------------------------------------------------------
# Reconciliation with the submitted Table S5 / historical JSONs (item 6)
# --------------------------------------------------------------------------------------------

def reconcile(s5, jsons, groups):
    """s5: DataFrame Class/Pair/Delta; jsons: {(enc, method, 'full'|'tumour'): dict}. Returns report dict."""
    groups = list(groups)
    full_labels = sorted(groups + ['NORMAL'])
    f1_checks, macro_checks, delta_checks, structure = [], [], [], []
    for (e, m, ctx), d in sorted(jsons.items()):
        labels = list(d['labels_order'])
        exp_labels = full_labels if ctx == 'full' else groups
        exp_n = 2778 if ctx == 'full' else N_QUERIES
        cm = np.asarray(d['confusion_matrix'], float)
        st = {'json': f'{e}:{m}:{ctx}', 'labels_ok': labels == exp_labels, 'n_slides': d['n_slides'],
              'n_slides_ok': d['n_slides'] == exp_n, 'cm_sum_ok': bool(cm.sum() == d['n_slides']),
              'top_k_ok': d['config'].get('top_k') == K,
              'tie_break_ok': d['config'].get('tie_break') == 'generous_majority',
              'dropped_labels': d['config'].get('dropped_labels')}
        st['pass'] = all(v for kk, v in st.items() if kk.endswith('_ok'))
        structure.append(st)
        f1 = f1_from_confusion(cm)
        pc = d['point_estimates']['per_class']
        for i, c in enumerate(labels):
            err = abs(f1[i] - pc[c]['f1'])
            f1_checks.append({'json': f'{e}:{m}:{ctx}', 'class': c, 'recomputed': float(f1[i]),
                              'saved': pc[c]['f1'], 'abs_err': float(err), 'pass': bool(err <= RECON_F1_TOL)})
        merr = abs(float(np.mean(f1)) - d['point_estimates']['macro_f1'])
        macro_checks.append({'json': f'{e}:{m}:{ctx}', 'recomputed': float(np.mean(f1)),
                             'saved': d['point_estimates']['macro_f1'], 'abs_err': merr,
                             'pass': bool(merr <= RECON_F1_TOL)})
    inv_e = {v: k for k, v in ENC_S5.items()}
    inv_m = {v: k for k, v in METHOD_S5.items()}
    for _, r in s5.iterrows():
        enc, meth = str(r['Pair']).split(' + ')
        e, m = inv_e[enc], inv_m[meth]
        full = jsons[(e, m, 'full')]['point_estimates']
        tum = jsons[(e, m, 'tumour')]['point_estimates']
        if r['Class'] == 'Overall':
            ff, ft = full['macro_f1'], tum['macro_f1']
        else:
            ff, ft = full['per_class'][r['Class']]['f1'], tum['per_class'][r['Class']]['f1']
        rec = submitted_error_reduction(ff, ft)
        err = abs(rec - float(r['Delta'])) if np.isfinite(rec) else np.inf
        delta_checks.append({'class': r['Class'], 'pair': r['Pair'], 'submitted': float(r['Delta']),
                             'recomputed': float(rec), 'abs_err': float(err),
                             'pass': bool(err <= RECON_DELTA_TOL)})
    counts_ok = len(delta_checks) == 60 and len(f1_checks) == 116 and len(macro_checks) == 8
    ok = (counts_ok and all(x['pass'] for x in delta_checks + f1_checks + macro_checks)
          and all(s['pass'] for s in structure))
    return {'pass': bool(ok), 'counts_ok': bool(counts_ok),
            'n_delta': len(delta_checks), 'n_per_class_f1': len(f1_checks), 'n_macro_f1': len(macro_checks),
            'max_delta_abs_err': max((x['abs_err'] for x in delta_checks), default=np.nan),
            'max_f1_abs_err': max((x['abs_err'] for x in f1_checks + macro_checks), default=np.nan),
            'formula': '(F1_tumour - F1_full) / (1 - F1_full); NaN if |1 - F1_full| < 1e-8 '
                       '(make_figure_3b.py compute_effects, metric=error_reduction)',
            'tolerances': {'delta': RECON_DELTA_TOL, 'f1': RECON_F1_TOL},
            'json_structure': structure, 'delta_cells': delta_checks, 'per_class_f1': f1_checks,
            'macro_f1': macro_checks}


# --------------------------------------------------------------------------------------------
# Loading the frozen packages
# --------------------------------------------------------------------------------------------

def _canon_budget(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ''
    return BUDGET_CANON[str(raw)]


def load_arms(enc, rl, qm, pop):
    """Validate one package and return {(method, arm, budget): arm-dict} aligned to population order."""
    n = len(pop)
    qidx = {int(s): i for i, s in enumerate(pop.slide_id)}
    truth = pop.diagnostic_group.to_numpy()
    patient = pop.patient_id.to_numpy()
    group_of = dict(zip(pop.slide_id.astype(int), pop.diagnostic_group))
    patient_of = dict(zip(pop.slide_id.astype(int), pop.patient_id))

    for name, df in (('ranked_lists', rl), ('query_metrics', qm)):
        raw = set(df.budget.dropna().astype(str))
        if raw != RAW_BUDGETS[enc]:
            raise GateError('budget_strings', f'{enc} {name}: stored budgets {sorted(raw)} != {sorted(RAW_BUDGETS[enc])}')
    rl = rl.assign(budget=rl.budget.map(_canon_budget))
    qm = qm.assign(budget=qm.budget.map(_canon_budget))
    q5 = qm[qm.k == K]
    expected_keys = {(m, a, b) for m in METHODS for a, b in ARMS[enc]}
    got_keys = set(map(tuple, q5[['method', 'arm', 'budget']].drop_duplicates().to_numpy().tolist()))
    if got_keys != expected_keys:
        raise GateError('arms', f'{enc}: query_metrics arms {sorted(got_keys)} != {sorted(expected_keys)}')
    rl_keys = set(map(tuple, rl[['method', 'arm', 'budget']].drop_duplicates().to_numpy().tolist()))
    if rl_keys != expected_keys:
        raise GateError('arms', f'{enc}: ranked_lists arms {sorted(rl_keys)} != {sorted(expected_keys)}')

    arms = {}
    rl_sorted = rl.sort_values(['method', 'arm', 'budget', 'query', 'rank'], kind='mergesort')
    rl_groups = {key: g for key, g in rl_sorted.groupby(['method', 'arm', 'budget'], sort=False)}
    for key in sorted(expected_keys):
        m, a, b = key
        q = q5[(q5.method == m) & (q5.arm == a) & (q5.budget == b)]
        if len(q) != n or q['query'].duplicated().any() or set(q['query'].astype(int)) != set(qidx):
            raise GateError('query_status', f'{enc} {key}: {len(q)} k={K} rows, not one per population query')
        bad_status = sorted(set(q.status) - set(ALLOWED_STATUS))
        if bad_status:
            raise GateError('query_status', f'{enc} {key}: unexpected status values {bad_status}')
        order = q['query'].astype(int).map(qidx).to_numpy()
        status = np.empty(n, dtype=object)
        status[order] = q.status.to_numpy()
        btie_qm = np.zeros(n, dtype=int)
        btie_qm[order] = q.boundary_tie.astype(int).to_numpy()
        chit = np.full(n, np.nan)
        chit[order] = q.conditional_hit.to_numpy(dtype=float)
        if not (np.array_equal(q.diagnostic_group.to_numpy(), truth[order])
                and np.array_equal(q.patient_id.astype(str).to_numpy(), patient[order].astype(str))):
            raise GateError('query_status', f'{enc} {key}: query_metrics truth/patient differ from the population')
        returned = status == 'returned'

        g = rl_groups[key]
        qs = g['query'].astype(int).to_numpy()
        if not set(qs) <= set(qidx):
            raise GateError('ranked_lists', f'{enc} {key}: ranked query not in the population')
        if set(qs) != {int(pop.slide_id.iat[i]) for i in np.flatnonzero(returned)}:
            raise GateError('ranked_lists', f'{enc} {key}: ranked queries != returned queries')
        nb = g['neighbour'].astype(int).to_numpy()
        ngrp = g['neighbour_group'].to_numpy()
        if not all(x in group_of for x in nb):
            raise GateError('ranked_lists', f'{enc} {key}: neighbour outside the population')
        if not np.array_equal(ngrp, np.array([group_of[x] for x in nb], dtype=object)):
            raise GateError('ranked_lists', f'{enc} {key}: neighbour_group differs from the population group')
        qtruth = np.array([group_of[x] for x in qs], dtype=object)
        if not np.array_equal(g['relevant'].astype(int).to_numpy(), (ngrp == qtruth).astype(int)):
            raise GateError('ranked_lists', f'{enc} {key}: relevant != (neighbour_group == truth)')
        if np.any(np.array([patient_of[x] for x in nb], dtype=object) == np.array([patient_of[x] for x in qs],
                                                                                    dtype=object)):
            raise GateError('ranked_lists', f'{enc} {key}: same-patient neighbour present')
        cuts = np.flatnonzero(np.diff(qs)) + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [len(qs)]])
        ranks = g['rank'].astype(int).to_numpy()
        dist = g['distance'].to_numpy(dtype=float)
        lists = {}
        hit_mismatch = 0
        tie_mismatch = 0
        for s, e_ in zip(starts, ends):
            sid = int(qs[s])
            r = ranks[s:e_]
            if not np.array_equal(r, np.arange(1, e_ - s + 1)) or (e_ - s) < K:
                raise GateError('ranked_lists', f'{enc} {key} query {sid}: ranks {r.tolist()} lack 1..{K}')
            d = dist[s:e_]
            if np.any(np.diff(d) < 0):
                raise GateError('ranked_lists', f'{enc} {key} query {sid}: distances not ascending')
            labs = ngrp[s:e_].tolist()
            i = qidx[sid]
            hit5 = float(any(x == truth[i] for x in labs[:K]))
            hit_mismatch += int(hit5 != chit[i])
            tie_mismatch += int(int(has_boundary_tie(d)) != btie_qm[i])
            lists[i] = (labs, d)
        if hit_mismatch:
            raise GateError('hit5_cross_check', f'{enc} {key}: {hit_mismatch} hit@5 mismatches with conditional_hit')
        if tie_mismatch:
            raise GateError('boundary_tie_agreement', f'{enc} {key}: {tie_mismatch} boundary_tie mismatches')
        if int(btie_qm[~returned].sum()):
            raise GateError('boundary_tie_agreement', f'{enc} {key}: boundary_tie set for non-returned queries')
        nonret = ~returned
        arms[key] = {'encoder': enc, 'method': m, 'arm': a, 'budget': b, 'status': status, 'nonret': nonret,
                     'lists': lists, 'boundary_tie': btie_qm,
                     'nonret_ids': {int(pop.slide_id.iat[i]) for i in np.flatnonzero(nonret)}}
    return arms


def arm_predictions(arm, truth, rule, plurality, legacy_generous, reverse_ties=False):
    """Predictions aligned to population order; NONE for non-returned queries (never voted)."""
    preds = []
    for i in range(len(truth)):
        if arm['nonret'][i]:
            preds.append(NONE_LABEL)
            continue
        labs, d = arm['lists'][i]
        if reverse_ties:
            labs, _ = reverse_boundary_block(labs, d)
        preds.append(vote(labs[:K], truth[i], rule, plurality, legacy_generous))
    return preds


# --------------------------------------------------------------------------------------------
# Contrasts
# --------------------------------------------------------------------------------------------

def contrast_specs():
    specs = []
    for e in ('uni2', 'virchow2'):
        for m in METHODS:
            specs.append({'encoder': e, 'method': m, 'design': 'legacy_tumour_only', 'arm': 'legacy_tumour_only',
                          'budget': 'legacy'})
    for m in METHODS:
        for design in ('query_only', 'both_filtered'):
            for b in ('5.0', '100.0'):
                specs.append({'encoder': 'hoptimus', 'method': m, 'design': design, 'arm': design, 'budget': b})
    for s in specs:
        s['contrast_id'] = f"{s['encoder']}:{s['method']}:{s['design']}:{s['budget']}-vs-all_tissue"
    return specs


def evaluate_contrast(W, T, PA, PF, inc):
    """Point and replicate F1 for both arms; returns dict of arrays (point: shape (C,), replicates (R, C))."""
    fA_pt, sup_pt = weighted_f1(np.ones((1, T.shape[0])), T, PA, inc)
    fF_pt, _ = weighted_f1(np.ones((1, T.shape[0])), T, PF, inc)
    out = {'fA_pt': fA_pt[0], 'fF_pt': fF_pt[0], 'sup_pt': sup_pt[0]}
    if W is not None:
        fA, sup = weighted_f1(W, T, PA, inc)
        fF, _ = weighted_f1(W, T, PF, inc)
        out.update(fA=fA, fF=fF, sup=sup)
    return out


def summarise_contrast(res, groups, n_reps):
    """Per-class and MACRO rows: estimates for both arms and contrast statistics."""
    C = len(groups)
    boot = 'fA' in res
    fA_pt, fF_pt, sup_pt = res['fA_pt'], res['fF_pt'], res['sup_pt']
    if np.any(sup_pt <= 0):
        raise GateError('point_support', 'a class has zero support at the point estimate')
    MA_pt, _ = macro_mean(fA_pt)
    MF_pt, _ = macro_mean(fF_pt)
    rows = []
    if boot:
        fA, fF, sup = res['fA'], res['fF'], res['sup']
        zero = sup <= 0
        MA, nexA = macro_mean(fA)
        MF, _ = macro_mean(fF)
        rer_c, und_c = rer_values(fF, fA)
        rer_m, und_m = rer_values(MF, MA)
        reps = {'MA': MA, 'MF': MF, 'rer_m': rer_m, 'dM': MF - MA}
    else:
        reps = None
    for j in range(C + 1):
        is_macro = j == C
        cls = MACRO if is_macro else groups[j]
        a_pt = float(MA_pt) if is_macro else float(fA_pt[j])
        f_pt = float(MF_pt) if is_macro else float(fF_pt[j])
        r_pt, u_pt = rer_values(f_pt, a_pt)
        row = {'class': cls, 'f1_all': a_pt, 'f1_filt': f_pt, 'delta_f1': f_pt - a_pt,
               'rer': float(r_pt), 'support': float(sup_pt.sum() if is_macro else sup_pt[j])}
        if boot:
            if is_macro:
                ra, rf, rr, ru, rz = MA, MF, rer_m, und_m, np.zeros(n_reps, bool)
                row['n_zero_support_reps'] = int(np.sum(nexA > 0))
            else:
                ra, rf, rr, ru, rz = fA[:, j], fF[:, j], rer_c[:, j], und_c[:, j], zero[:, j]
                row['n_zero_support_reps'] = int(rz.sum())
            row['f1_all_ci'] = percentile_ci(ra)
            row['f1_filt_ci'] = percentile_ci(rf)
            row['delta_ci'] = percentile_ci(rf - ra)
            row.update(rer_interval(bool(u_pt), rr, ru, rz, is_macro))
        else:
            row['n_zero_support_reps'] = np.nan
            row['f1_all_ci'] = row['f1_filt_ci'] = row['delta_ci'] = (np.nan, np.nan)
            row.update(rer_ci_low=np.nan, rer_ci_high=np.nan,
                       rer_ci_status='undefined_point' if u_pt else 'no_bootstrap', n_undefined_reps=np.nan)
        rows.append(row)
    return rows, reps


# --------------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------------

def _csv_bytes(df):
    return df.to_csv(index=False, lineterminator='\n').encode()


def _write(out, name, data):
    path = os.path.join(out, name)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def _json_bytes(obj):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return sorted(o)
        raise TypeError(type(o))
    return (json.dumps(obj, indent=1, sort_keys=False, default=default, allow_nan=True) + '\n').encode()


def write_sha256sums(out):
    names = sorted(n for n in os.listdir(out) if n != 'SHA256SUMS.txt' and os.path.isfile(os.path.join(out, n)))
    lines = [f'{sha256_file(os.path.join(out, n))[0]}  {n}\n' for n in names]
    _write(out, 'SHA256SUMS.txt', ''.join(lines).encode())


def write_failed(out, gate, reason, detail=None, bindings=None):
    obj = {'status': 'FAILED', 'gate': gate, 'reason': reason, 'detail': detail,
           'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'lsf_job': os.environ.get('LSB_JOBID'), 'host': socket.gethostname(), 'input_bindings': bindings}
    _write(out, 'FAILED.json', _json_bytes(obj))
    write_sha256sums(out)


# --------------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------------

def input_paths(repo, comet_root, prosper_out):
    p = {'population': os.path.join(prosper_out, 'r2_1_mosaic_cpu_20261003', 'attempt_001',
                                    'retrieval_population.csv')}
    for enc, parts in PACKAGE_DIRS.items():
        d = os.path.join(prosper_out, *parts)
        p[f'ranked_lists:{enc}'] = os.path.join(d, 'ranked_lists.csv.gz')
        p[f'query_metrics:{enc}'] = os.path.join(d, 'query_metrics.csv.gz')
        p[f'patient_draws:{enc}'] = os.path.join(d, 'patient_draws.npy')
        p[f'summary:{enc}'] = os.path.join(d, 'summary.json')
    for e in ('uni2', 'virchow2'):
        for m in METHODS:
            for ctx, suffix in (('full', ''), ('tumour', '_tumor_only')):
                p[f'hist:{e}:{m}:{ctx}'] = os.path.join(
                    comet_root, f'comet-{e}', 'confidence_interval_tie_majority',
                    f'slide_level_CI_{e}_{m}_top_5{suffix}_no_same_patient.json')
    p['table_s5'] = os.path.join(repo, 'docs', 'prosper_submission', 'supplementary', 'supplementary_tables',
                                 'Table_S5_tumor_only_analysis.csv')
    p['vote_module'] = os.path.join(repo, 'scripts', 'r2_3_internal_failure_tables.py')
    return p


def hash_inputs(paths, pins=PIN):
    """Read every input once and hash those bytes; return (bindings, mismatches, data).

    All later parsing uses `data` (the hashed bytes); no input path is re-opened. Missing files are mismatches.
    """
    bindings, bad, data = {}, {}, {}
    for role, path in paths.items():
        rec = {'path': path, 'pinned_sha256': pins.get(role)}
        try:
            with open(path, 'rb') as f:
                b = f.read()
            data[role] = b
            rec.update(sha256=sha256_bytes(b), size=len(b))
        except OSError as ex:
            rec.update(sha256=None, size=None, error=f'{type(ex).__name__}: {ex}')
        rec['match'] = rec['sha256'] is not None and rec['sha256'] == rec['pinned_sha256']
        if not rec['match']:
            bad[role] = {'path': path, 'observed': rec['sha256'], 'pinned': rec['pinned_sha256'],
                         'error': rec.get('error')}
        bindings[role] = rec
    return bindings, bad, data


def run(repo, comet_root, out, bootstrap=True):
    """Execute the package into `out`. Raises GateError on any gate failure."""
    gates = {}
    prosper_out = os.environ.get('PROSPER_OUT') or os.path.join(repo, 'prosper_revision_output')
    paths = input_paths(repo, comet_root, prosper_out)
    bindings, bad, data = hash_inputs(paths)
    script_path = os.path.abspath(__file__)
    sh, ss = sha256_file(script_path)
    state = {'bindings': {'inputs': bindings, 'script': {'path': script_path, 'sha256': sh, 'size': ss},
                          'prosper_out': prosper_out,
                          'formula_reference_unpinned': _formula_reference(comet_root)}}
    run.state = state
    if bad:
        raise GateError('input_hashes', f'{len(bad)} input(s) missing or differ from the pinned sha256', bad)
    gates['input_hashes'] = f'{len(bindings)} inputs match pins'

    # Vote functions: the bytes read and hashed above are executed; the path is not re-opened (item 13).
    plurality, legacy_generous, vinfo = load_vote_functions(paths['vote_module'], source=data['vote_module'])
    gates['vote_module'] = 'pinned bytes executed via spec_from_loader'

    pop = pd.read_csv(io.BytesIO(data['population']))
    if not (len(pop) == N_QUERIES and pop.slide_id.is_unique and pop.patient_id.nunique() == N_PATIENTS
            and pop.diagnostic_group.nunique() == N_GROUPS):
        raise GateError('population', 'population is not 2,604 unique slides / 1,800 patients / 14 groups')
    groups = sorted(pop.diagnostic_group.unique())
    truth = pop.diagnostic_group.to_numpy()
    patients_sorted = sorted(pop.patient_id.unique())
    gates['population'] = '2604 slides / 1800 patients / 14 groups'

    # Draws (item 12)
    summaries, draws = {}, None
    for enc in PACKAGE_DIRS:
        summaries[enc] = json.loads(data[f'summary:{enc}'])
        d = np.load(io.BytesIO(data[f'patient_draws:{enc}']), allow_pickle=False)
        if draws is None:
            draws = d
        elif not np.array_equal(draws, d):
            raise GateError('draws_identical', f'{enc} patient_draws.npy differs')
    check_draws(draws, patients_sorted, summaries)
    gates['draws_binding'] = 'shape (5000, 1800) int64, row sums 1800, columns = sorted unique patient_id = ' \
                             'patient_draws_patients in all three summary.json; three copies identical'
    for enc, s in summaries.items():
        state['bindings']['inputs'][f'summary:{enc}']['patient_draws_patients_ok'] = True

    # Packages
    arms = {}
    for enc in PACKAGE_DIRS:
        rl = pd.read_csv(io.BytesIO(data[f'ranked_lists:{enc}']), compression='gzip', dtype={'budget': str})
        qm = pd.read_csv(io.BytesIO(data[f'query_metrics:{enc}']), compression='gzip', dtype={'budget': str})
        for key, a in load_arms(enc, rl, qm, pop).items():
            arms[(enc,) + key] = a
    observed = {}
    for (enc, m, a, b), arm in arms.items():
        observed[(enc, m, a, b)] = arm['nonret_ids']
    expected = {(enc, m, a, b): EXPECTED_NONRETURN[(enc, a, b)] for (enc, m, a, b) in arms}
    check_nonreturn_sets(observed, expected)
    sid_truth = dict(zip(pop.slide_id.astype(int), pop.diagnostic_group))
    for sid, g in EXPECTED_NONRETURN_TRUTH.items():
        if sid_truth.get(sid) != g:
            raise GateError('nonreturn_sets', f'query {sid} truth {sid_truth.get(sid)} != {g}')
    gates['query_status'] = f'{len(arms)} arms x 2604 k=5 rows; status in {list(ALLOWED_STATUS)}'
    gates['nonreturn_sets'] = 'exact contract sets for every package/arm/method/budget'
    gates['ranked_list_integrity'] = 'ranks 1..depth (>=5) per returned query; neighbours in population with ' \
                                     'matching group; relevant == group match; no same-patient neighbour'
    gates['hit5_cross_check'] = 'hit@5 from ranked_lists == conditional_hit for every returned query'
    gates['boundary_tie_agreement'] = 'd5 == d6 detection == query_metrics boundary_tie'

    # Predictions (original and, for every arm, tie-reversed; reversal asserts containment).
    preds, preds_rev = {}, {}
    for key, arm in arms.items():
        for rule in RULES:
            preds[key + (rule,)] = arm_predictions(arm, truth, rule, plurality, legacy_generous)
            preds_rev[key + (rule,)] = arm_predictions(arm, truth, rule, plurality, legacy_generous, True)
    gates['tie_block_containment'] = 'every k=5 boundary tie block lies within the stored top-10'

    # Reconciliation (item 6)
    s5 = pd.read_csv(io.BytesIO(data['table_s5']))
    jsons = {}
    for e in ('uni2', 'virchow2'):
        for m in METHODS:
            for ctx in ('full', 'tumour'):
                jsons[(e, m, ctx)] = json.loads(data[f'hist:{e}:{m}:{ctx}'])
    recon = reconcile(s5, jsons, groups)
    s5_map = {}
    for _, r in s5.iterrows():
        enc, meth = str(r['Pair']).split(' + ')
        e = {v: k for k, v in ENC_S5.items()}[enc]
        m = {v: k for k, v in METHOD_S5.items()}[meth]
        s5_map[(e, m, MACRO if r['Class'] == 'Overall' else r['Class'])] = float(r['Delta'])

    # Lineage (item 9)
    lineage, lineage_fail = lineage_check(arms, preds, preds_rev, jsons, groups, truth)

    outputs = {'reconciliation.json': _json_bytes(recon), 'lineage_check.json': _json_bytes(lineage)}
    if not recon['pass']:
        _flush(out, outputs)
        raise GateError('reconciliation', 'Table S5 / historical JSON reconciliation failed',
                        {'max_delta_abs_err': recon['max_delta_abs_err'], 'max_f1_abs_err': recon['max_f1_abs_err'],
                         'counts_ok': recon['counts_ok']})
    gates['reconciliation'] = (f"60 Delta cells <= {RECON_DELTA_TOL}; 116 per-class + 8 macro F1 <= {RECON_F1_TOL}")
    if lineage_fail:
        _flush(out, outputs)
        raise GateError('lineage_ot_expectation', 'OT lineage differences outside the pre-registered expectation',
                        lineage_fail)
    gates['lineage_ot_expectation'] = 'OT differences equal the pre-registered non-return truth->truth cells'

    # Bootstrap weights
    W = None
    if bootstrap:
        pcol = {p: i for i, p in enumerate(patients_sorted)}
        qcol = np.array([pcol[p] for p in pop.patient_id])
        W = draws[:, qcol].astype(np.float64)
    gi = {g: i for i, g in enumerate(groups)}
    T = np.zeros((len(pop), len(groups)))
    T[np.arange(len(pop)), [gi[t] for t in truth]] = 1.0
    PM = {k: prediction_matrix(v, groups) for k, v in preds.items()}
    PM_rev = {k: prediction_matrix(v, groups) for k, v in preds_rev.items()}

    est_rows, con_rows, pq_rows, tie_rows = [], [], [], []
    rep_store = {}
    specs = contrast_specs()
    for sp in specs:
        e, m = sp['encoder'], sp['method']
        kA = (e, m, 'all_tissue', '')
        kF = (e, m, sp['arm'], sp['budget'])
        armA, armF = arms[kA], arms[kF]
        for rule in RULES:
            for handling in HANDLINGS:
                inc = inclusion_mask(handling, armA['nonret'], armF['nonret'])
                res = evaluate_contrast(W, T, PM[kA + (rule,)], PM[kF + (rule,)], inc)
                rows, reps = summarise_contrast(res, groups, N_REPS)
                rep_store[(sp['contrast_id'], rule, handling)] = reps
                n_inc = int(inc.sum())
                incb = inc > 0
                for side, arm, key in (('all', armA, kA), ('filt', armF, kF)):
                    for r in rows:
                        est_rows.append({
                            'encoder': e, 'method': m, 'arm': arm['arm'], 'design': sp['design'],
                            'budget': arm['budget'], 'rule': rule, 'handling': handling,
                            'contrast_id': sp['contrast_id'], 'class': r['class'],
                            'f1': r['f1_all'] if side == 'all' else r['f1_filt'],
                            'ci_low': (r['f1_all_ci'] if side == 'all' else r['f1_filt_ci'])[0],
                            'ci_high': (r['f1_all_ci'] if side == 'all' else r['f1_filt_ci'])[1],
                            'support': r['support'], 'n_queries': n_inc,
                            'n_returned': int((~arm['nonret'] & incb).sum()),
                            'n_nonreturned': int((arm['nonret'] & incb).sum()),
                            'n_zero_support_reps': r['n_zero_support_reps'], 'quantile_method': QUANTILE_METHOD})
                    for i in np.flatnonzero(incb):
                        pq_rows.append((int(pop.slide_id.iat[i]), truth[i], e, m, arm['arm'], sp['design'],
                                        arm['budget'], rule, handling, sp['contrast_id'], arm['status'][i],
                                        int(arm['boundary_tie'][i]), preds[key + (rule,)][i]))
                for r in rows:
                    route_a = e in ENC_S5
                    con_rows.append({
                        'encoder': e, 'method': m, 'arm': sp['arm'], 'design': sp['design'],
                        'budget': sp['budget'], 'rule': rule, 'handling': handling,
                        'contrast_id': sp['contrast_id'], 'class': r['class'],
                        'f1_all': r['f1_all'], 'f1_filt': r['f1_filt'], 'delta_f1': r['delta_f1'],
                        'delta_ci_low': r['delta_ci'][0], 'delta_ci_high': r['delta_ci'][1],
                        'rer': r['rer'], 'rer_ci_low': r['rer_ci_low'], 'rer_ci_high': r['rer_ci_high'],
                        'rer_ci_status': r['rer_ci_status'], 'n_undefined_reps': r['n_undefined_reps'],
                        'n_zero_support_reps': r['n_zero_support_reps'], 'support': r['support'],
                        'n_queries': n_inc,
                        'n_nonreturned_filt': int((armF['nonret'] & incb).sum()),
                        'submitted_delta': s5_map[(e, m, r['class'])] if route_a else np.nan,
                        'submitted_population_note': SUBMITTED_NOTE if route_a else '',
                        'interval_label': ('prespecified_item8 (unadjusted)'
                                           if r['class'] in PRESPECIFIED_GROUPS + (MACRO,)
                                           else 'unadjusted, descriptive'),
                        'quantile_method': QUANTILE_METHOD})
                if m == 'yottixel':
                    tie_rows.extend(tie_sensitivity_rows(sp, rule, handling, inc, T, groups,
                                                         PM, PM_rev, kA, kF, rows))
    gates['point_support'] = 'every class has positive support at the point estimate in every contrast'

    method_rows = method_contrasts(specs, rep_store, con_rows, bootstrap)

    est = pd.DataFrame(est_rows)
    con = pd.DataFrame(con_rows)
    mc = pd.DataFrame(method_rows)
    ts = pd.DataFrame(tie_rows)
    tc = pd.DataFrame(tie_count_rows(arms, preds, preds_rev))
    pq = pd.DataFrame(pq_rows, columns=['query', 'truth', 'encoder', 'method', 'arm', 'design', 'budget', 'rule',
                                        'handling', 'contrast_id', 'status', 'boundary_tie', 'prediction'])
    outputs.update({
        'estimates.csv': _csv_bytes(est), 'contrasts.csv': _csv_bytes(con), 'method_contrasts.csv': _csv_bytes(mc),
        'tie_sensitivity.csv': _csv_bytes(ts), 'tie_counts.csv': _csv_bytes(tc),
        'per_query_predictions.csv.gz': gzip.compress(_csv_bytes(pq), mtime=0),
    })
    state['bindings']['versions'] = {'python': platform.python_version(), 'numpy': np.__version__,
                                     'pandas': pd.__version__}
    state['bindings']['vote_module_exec'] = vinfo
    outputs['input_bindings.json'] = _json_bytes(state['bindings'])
    _flush(out, outputs)
    complete = {
        'status': 'COMPLETE', 'contract': 'ADR0021 B(c) candidate 002', 'bootstrap': bool(bootstrap),
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'lsf_job': os.environ.get('LSB_JOBID'), 'host': socket.gethostname(), 'argv': sys.argv,
        'gates': gates, 'n_contrasts': len(specs), 'rules': list(RULES), 'handlings': list(HANDLINGS),
        'undefined_threshold': UNDEFINED_THRESHOLD, 'max_undefined_reps_per_class': max_undefined_reps(N_REPS),
        'outputs': {n: sha256_bytes(b) for n, b in sorted(outputs.items())},
        'notes': ['legacy_generous_ORACLE is an oracle rule (uses the truth label to break ties): reconciliation and '
                  'like-for-like with Table S5 only', 'all intervals are unadjusted, descriptive; "interval excludes '
                  'zero" statements are confined to MACRO and the item-8(d) groups',
                  'point estimates from the unweighted population; intervals 95 % linear percentile over the saved '
                  'whole-patient draws']}
    _write(out, 'COMPLETE.json', _json_bytes(complete))
    write_sha256sums(out)
    return {'gates': gates, 'n_estimates': len(est), 'n_contrasts_rows': len(con), 'n_method_rows': len(mc),
            'n_tie_rows': len(ts), 'n_per_query': len(pq)}


def _flush(out, outputs):
    for name, data in outputs.items():
        _write(out, name, data)


def _formula_reference(comet_root):
    path = os.path.join(comet_root, 'comet-visualization', 'src', 'make_figure_3b.py')
    try:
        h, s = sha256_file(path)
    except OSError as ex:
        return {'path': path, 'error': str(ex)}
    return {'path': path, 'sha256': h, 'size': s,
            'note': 'formula re-implemented in submitted_error_reduction (eps 1e-8); recorded, not gated'}


def lineage_check(arms, preds, preds_rev, jsons, groups, truth):
    """Item 9: legacy_generous on legacy tumour-only arms vs historical tumour-only confusion matrices."""
    gi = {g: i for i, g in enumerate(groups)}
    report = {'description': 'historical tumour-only JSON confusion_matrix (H) vs reproduced matrix (C) from '
                             'route-(a) legacy_tumour_only lists with legacy_generous_ORACLE over returned queries; '
                             'diff = H - C', 'disclosure_line': DISCLOSURE_LINE, 'arms': []}
    fail = {}
    for e in ('uni2', 'virchow2'):
        for m in METHODS:
            key = (e, m, 'legacy_tumour_only', 'legacy')
            arm = arms[key]
            H = np.asarray(jsons[(e, m, 'tumour')]['confusion_matrix'], int)
            if list(jsons[(e, m, 'tumour')]['labels_order']) != list(groups):
                fail[f'{e}:{m}'] = 'historical labels_order differs from population groups'
                continue
            C = np.zeros((len(groups), len(groups)), int)
            p = preds[key + ('legacy_generous_ORACLE',)]
            for i in range(len(truth)):
                if arm['nonret'][i]:
                    continue
                C[gi[truth[i]], gi[p[i]]] += 1
            D = H - C
            cells = [{'truth': groups[i], 'pred': groups[j], 'historical': int(H[i, j]), 'reproduced': int(C[i, j]),
                      'diff': int(D[i, j])} for i, j in np.argwhere(D != 0)]
            ch_g = sum(a != b for a, b in zip(p, preds_rev[key + ('legacy_generous_ORACLE',)]))
            ch_p = sum(a != b for a, b in zip(preds[key + ('plurality',)], preds_rev[key + ('plurality',)]))
            rec = {'encoder': e, 'method': m, 'arm': 'legacy_tumour_only', 'budget': 'legacy',
                   'rule': 'legacy_generous_ORACLE', 'historical_matrix': H, 'reproduced_matrix': C,
                   'labels_order': list(groups), 'cell_differences': cells, 'n_cells_differing': len(cells),
                   'sum_abs_diff': int(np.abs(D).sum()), 'nonreturned_queries': sorted(arm['nonret_ids']),
                   'boundary_tie_queries': int(arm['boundary_tie'].sum()),
                   'tie_reversal_prediction_changes': {'legacy_generous_ORACLE': int(ch_g), 'plurality': int(ch_p)}}
            if m == 'ot_cosine':
                exp = LINEAGE_OT_EXPECTATION[e]
                obs = {(c['truth'], c['pred']): c['diff'] for c in cells}
                rec['preregistered_expectation'] = [{'truth': t, 'pred': q, 'diff': v} for (t, q), v in exp.items()]
                rec['expectation_met'] = obs == exp
                if obs != exp:
                    fail[f'{e}:{m}'] = {'observed': [{'truth': t, 'pred': q, 'diff': v} for (t, q), v in obs.items()],
                                        'expected': rec['preregistered_expectation']}
            else:
                rec['expectation'] = 'differences reported as counts beside boundary-tie and tie-reversal counts'
            report['arms'].append(rec)
    report['ot_expectation_pass'] = not fail
    return report, fail


def tie_sensitivity_rows(sp, rule, handling, inc, T, groups, PM, PM_rev, kA, kF, rows):
    """Item 10: point-estimate F1/RER/dF1 with each boundary tie block reversed in both arms."""
    ones = np.ones((1, T.shape[0]))
    fA, _ = weighted_f1(ones, T, PM_rev[kA + (rule,)], inc)
    fF, _ = weighted_f1(ones, T, PM_rev[kF + (rule,)], inc)
    fA, fF = fA[0], fF[0]
    MA, _ = macro_mean(fA)
    MF, _ = macro_mean(fF)
    incb = inc > 0
    chA = int(np.sum(np.any(PM_rev[kA + (rule,)] != PM[kA + (rule,)], axis=1) & incb))
    chF = int(np.sum(np.any(PM_rev[kF + (rule,)] != PM[kF + (rule,)], axis=1) & incb))
    out = []
    for j, r in enumerate(rows):
        is_macro = r['class'] == MACRO
        a = float(MA) if is_macro else float(fA[j])
        f = float(MF) if is_macro else float(fF[j])
        rr, _ = rer_values(f, a)
        out.append({'encoder': sp['encoder'], 'method': sp['method'], 'arm': sp['arm'], 'design': sp['design'],
                    'budget': sp['budget'], 'rule': rule, 'handling': handling, 'contrast_id': sp['contrast_id'],
                    'class': r['class'], 'f1_all': r['f1_all'], 'f1_filt': r['f1_filt'], 'delta_f1': r['delta_f1'],
                    'rer': r['rer'], 'f1_all_tie_reversed': a, 'f1_filt_tie_reversed': f,
                    'delta_f1_tie_reversed': f - a, 'rer_tie_reversed': float(rr),
                    'delta_f1_shift': (f - a) - r['delta_f1'], 'rer_shift': float(rr) - r['rer'],
                    'n_prediction_changes_all': chA, 'n_prediction_changes_filt': chF})
    return out


def tie_count_rows(arms, preds, preds_rev):
    rows = []
    for key in sorted(arms):
        arm = arms[key]
        e, m, a, b = key
        rows.append({'encoder': e, 'method': m, 'arm': a, 'budget': b,
                     'returned_queries': int((~arm['nonret']).sum()),
                     'boundary_tie_queries': int(arm['boundary_tie'].sum()),
                     'tie_reversal_changes_plurality': int(sum(
                         x != y for x, y in zip(preds[key + ('plurality',)], preds_rev[key + ('plurality',)]))),
                     'tie_reversal_changes_legacy_generous_ORACLE': int(sum(
                         x != y for x, y in zip(preds[key + ('legacy_generous_ORACLE',)],
                                                preds_rev[key + ('legacy_generous_ORACLE',)])))})
    return rows


def method_contrasts(specs, rep_store, con_rows, bootstrap):
    """Item 8(c): within encoder/design/budget/rule/handling, macro RER_YX - RER_OT and dF1_YX - dF1_OT."""
    macro = {(r['contrast_id'], r['rule'], r['handling']): r for r in con_rows if r['class'] == MACRO}
    by = {}
    for sp in specs:
        by.setdefault((sp['encoder'], sp['design'], sp['budget']), {})[sp['method']] = sp['contrast_id']
    rows = []
    for (e, design, b), d in by.items():
        for rule in RULES:
            for handling in HANDLINGS:
                ry = macro[(d['yottixel'], rule, handling)]
                ro = macro[(d['ot_cosine'], rule, handling)]
                row = {'encoder': e, 'method': 'yottixel_minus_ot_cosine', 'arm': design, 'design': design,
                       'budget': b, 'rule': rule, 'handling': handling,
                       'contrast_id': f'{e}:yottixel-minus-ot_cosine:{design}:{b}',
                       'rer_yx': ry['rer'], 'rer_ot': ro['rer'], 'rer_yx_minus_ot': ry['rer'] - ro['rer'],
                       'delta_f1_yx': ry['delta_f1'], 'delta_f1_ot': ro['delta_f1'],
                       'delta_f1_yx_minus_ot': ry['delta_f1'] - ro['delta_f1']}
                if bootstrap:
                    py = rep_store[(d['yottixel'], rule, handling)]
                    po = rep_store[(d['ot_cosine'], rule, handling)]
                    diff = py['rer_m'] - po['rer_m']
                    n_und = int(np.sum(~np.isfinite(diff)))
                    row['rer_yx_minus_ot_ci_low'], row['rer_yx_minus_ot_ci_high'] = percentile_ci(diff)
                    row['n_undefined_reps'] = n_und
                    dd = py['dM'] - po['dM']
                    row['delta_f1_yx_minus_ot_ci_low'], row['delta_f1_yx_minus_ot_ci_high'] = percentile_ci(dd)
                else:
                    row.update(rer_yx_minus_ot_ci_low=np.nan, rer_yx_minus_ot_ci_high=np.nan, n_undefined_reps=np.nan,
                               delta_f1_yx_minus_ot_ci_low=np.nan, delta_f1_yx_minus_ot_ci_high=np.nan)
                row['interval_label'] = ('prespecified_item8c (unadjusted)' if e in ENC_S5
                                         else 'descriptive (item 8c, H-Optimus)')
                row['quantile_method'] = QUANTILE_METHOD
                rows.append(row)
    return rows


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True, help='output directory (must not exist or be empty)')
    ap.add_argument('--repo', default=here, help='repository root (default: parent of scripts/)')
    ap.add_argument('--comet-root', default=None, help='historical comet-feature-gen root (default REPO/../'
                                                       'comet-feature-gen)')
    ap.add_argument('--no-bootstrap', action='store_true', help='tests only: skip the bootstrap intervals')
    args = ap.parse_args(argv)
    repo = os.path.abspath(args.repo)
    comet_root = os.path.abspath(args.comet_root or os.path.join(repo, '..', 'comet-feature-gen'))
    out = os.path.abspath(args.out)
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        print(f'error: --out {out} exists and is not an empty directory', file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    run.state = {'bindings': None}
    try:
        summary = run(repo, comet_root, out, bootstrap=not args.no_bootstrap)
    except GateError as ex:
        write_failed(out, ex.gate, ex.reason, ex.detail, (getattr(run, 'state', {}) or {}).get('bindings'))
        print(f'FAILED gate={ex.gate}: {ex.reason} (see {out}/FAILED.json)')
        return 1
    except Exception as ex:  # any unexpected error is also a failed run, never a silent pass
        write_failed(out, 'unexpected_error', f'{type(ex).__name__}: {ex}', traceback.format_exc(),
                     (getattr(run, 'state', {}) or {}).get('bindings'))
        print(f'FAILED unexpected {type(ex).__name__}: {ex} (see {out}/FAILED.json)')
        return 1
    print(f'COMPLETE: {len(summary["gates"])} gates passed; estimates={summary["n_estimates"]} '
          f'contrast_rows={summary["n_contrasts_rows"]} method_rows={summary["n_method_rows"]} '
          f'tie_rows={summary["n_tie_rows"]} per_query_rows={summary["n_per_query"]} -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
