#!/usr/bin/env python3
"""R2.1 route (b) UNI2: B(c) fair-design top-5 vote F1 RER driver (adapter A-5 of R2.1-ROUTE-B-UNI2).

Specification: OUT/r2_1_route_b_uni2_20261013/contract_candidate_002/SPECIFICATION.md §3 A-5, §4 step 6, §4 step 7 R-0,
§12.7/12.16/12.17 (LEAD_RESOLUTIONS_002 + ADDENDUM unchanged for this adapter).

The frozen `scripts/r2_1_filter_rer.py` (sha256 bf29e631...) is read once, its full sha256 is verified and those
same bytes are executed into a PRIVATE module instance (never registered as `r2_1_filter_rer`). Its functions are
used unchanged. This driver supplies its own PIN, ARMS (and the RAW_BUDGETS read by the frozen `load_arms`),
EXPECTED_NONRETURN and input paths, as private globals of that instance, and its own contrast loop because the
frozen `run()`/`contrast_specs()`/`method_contrasts()` are keyed on the route-(a) encoders (`ENC_S5`).

Modes
  run          route (b) UNI2: 8 contrasts (ot_cosine/yottixel x query_only/both_filtered x 5.0/100.0) vs the UNI2
               all_tissue arm -> --out (ROOT/rer_run_002 under AMENDMENT_002; 480 contrasts.csv rows, run_001's
               27 columns, submitted_delta NaN and submitted_population_note '' for every row; COMPLETE.json
               source_run "route_b_uni2",
               amendment "002"). AMENDMENT_002: a k=5 boundary tie block reaching the last stored rank is resolved
               from the frozen _no_same_patient.pkl matrix row (TieBlockResolver); evidence in
               tie_block_matrix_resolution.json, matrices bound under input_bindings.json tie_block_matrices.
  r0           R-0 regression: the same driver in the H-Optimus-1 configuration (frozen PIN/EXPECTED_NONRETURN,
               accepted r2_1_filter_20261003/evaluation_qf) into ROOT/fidelity/r0/<dir>; its contrasts.csv and
               estimates.csv must equal run_001's hoptimus rows byte-identically after selecting those rows with
               the same CSV writer. The AMENDMENT_002 matrix path is disabled here (the frozen rule applies).
  derive-pins  writes the route-(b) pin file from the mask/bank census (`_filter_summary.csv`, own_fold_query == 1
               & written == 0 -> EXPECTED_NONRETURN; independent of the evaluator) plus the evaluation_qf
               summary.json output hashes. Reads small files only.

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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.dirname(HERE)
FROZEN_RER_SHA256 = 'b292ca107e5841793b6182aebc4f8d9db6c0730d24962a8bea6f4339eb404367'
FROZEN_RER_PATH = os.path.join(HERE, 'r2_1_filter_rer.py')
ROOT_NAME = 'r2_1_route_b_uni2_20261013'
SOURCE_RUN = 'route_b_uni2'
R0_SOURCE_RUN = 'r0_hoptimus_regression'

# §3 A-5 (review 001 M2 verbatim): exactly run_001's 27 columns in run_001's order.
CONTRAST_COLUMNS = [
    'encoder', 'method', 'arm', 'design', 'budget', 'rule', 'handling', 'contrast_id', 'class', 'f1_all',
    'f1_filt', 'delta_f1', 'delta_ci_low', 'delta_ci_high', 'rer', 'rer_ci_low', 'rer_ci_high', 'rer_ci_status',
    'n_undefined_reps', 'n_zero_support_reps', 'support', 'n_queries', 'n_nonreturned_filt', 'submitted_delta',
    'submitted_population_note', 'interval_label', 'quantile_method']
ESTIMATE_COLUMNS = ['encoder', 'method', 'arm', 'design', 'budget', 'rule', 'handling', 'contrast_id', 'class', 'f1',
                    'ci_low', 'ci_high', 'support', 'n_queries', 'n_returned', 'n_nonreturned', 'n_zero_support_reps',
                    'quantile_method']
PQ_COLUMNS = ['query', 'truth', 'encoder', 'method', 'arm', 'design', 'budget', 'rule', 'handling', 'contrast_id',
              'status', 'boundary_tie', 'prediction']

DESIGNS = ('query_only', 'both_filtered')
BUDGETS = ('5.0', '100.0')
BUDGET_DIR = {'5.0': 'budget5', '100.0': 'budget100'}
N_FOLDS = 5

# Pins fixed by the specification (§2): population, saved whole-patient draws (F-4), run_001 outputs (R-0).
POPULATION_SHA256 = 'c7cae1f0d855952e84e85aa90c7ef7d8a622086273c113285655c96be946d97b'
DRAWS_SHA256 = 'cfaed70d0c11084fb72b4dd7c787313b5da9d6e47f374803d2671b647886f350'
RUN001_CONTRASTS_SHA256 = '2ff09ac355d615c4cfff48ba71d86322f198f8ea39e24d9264fa72a073b82507'
RUN001_ESTIMATES_SHA256 = '549bb456d85ee909e2aeb4aa703f86181e8849b4198fb149edc7085b0f37fa25'
RUN001_DIAGNOSTIC_SHA256 = {  # R-0 diagnostics only (not the specification gate)
    'method_contrasts.csv': 'ea3e83b336c979dda7e86d7b9080ef41f3e9b66240cad627444e2ac6fa03e354',
    'tie_counts.csv': 'f5f697ae46793edaec0c66311d8a804cd3fb106cb222aad624ab26f2618153ef',
    'tie_sensitivity.csv': 'a1e2ced93f7eaded33059c728515d9910b94a455175ec9f7013a55a1df1587c8',
}
PIN_KEYS = ('ranked_lists', 'query_metrics', 'summary')

# AMENDMENT_002 (c1659ef6...): containment evidence from the frozen no-same-patient matrix row.
AMENDMENT = '002'
SENTINEL = 1e6
RESOLUTION_FILE = 'tie_block_matrix_resolution.json'
QF_MATRIX_NAME = 'combined_similarity_matrix_r2_1_hopt_filter_qf_budget{b}_{d}_{m}_no_same_patient.pkl'
DESIGN_TOKEN = {'query_only': 'query', 'both_filtered': 'both'}

NOT_APPLICABLE = {'not_applicable': 'route (b) has no submitted counterpart'}
ROUTEB_METHOD_INTERVAL_LABEL = 'descriptive (item 8c, route (b) UNI2)'
README_TEXT = """# R2.1 route (b) UNI2 - B(c) fair-design RER (Step 6 output; `rer_run_002` under AMENDMENT_002)

Source/route statement (contract R2.1-ROUTE-B-UNI2 candidate 002, §3 A-5 / §4 step 6):

- These rows are the UNI2 **route (b)** RER: new UNI2 tumour heads (5-fold patient-held-out, budgets 5 % and
  100 %), designs `query_only` and `both_filtered`, each versus the UNI2 `all_tissue` arm of the same method.
- Route identity is carried by this directory, by `COMPLETE.json` field `source_run: "route_b_uni2"` and by the
  `contrast_id` values `uni2:{method}:{design}:{budget}-vs-all_tissue`. There is no `source_run` column.
- `submitted_delta` is empty (NaN) and `submitted_population_note` is '' for every row: route (b) has no
  submitted counterpart.
- When these rows are combined with `r2_1_filter_rer_20261012/run_001`, select on `contrast_id`/design, never on
  `encoder` alone (run_001 also contains route-(a) `uni2:...:legacy_tumour_only:legacy-vs-all_tissue` rows).
- `reconciliation.json` and `lineage_check.json` are route-(a)-specific and are written as not applicable.
- Tie sensitivity (AMENDMENT_002): a k=5 boundary tie block that reaches the last stored rank is reversed only if
  the frozen `_no_same_patient.pkl` matrix row shows it ends exactly there (n_lt == lo, n_eq == hi - lo + 1);
  every such row is listed in `tie_block_matrix_resolution.json`, and the matrices read are bound in
  `input_bindings.json` under `tie_block_matrices`. `rer_run_001` is the preserved failed attempt.
- Driver: `scripts/r2_1_filter_rer_routeb.py`, executing the sha-verified bytes of the frozen
  `scripts/r2_1_filter_rer.py`; hashes in `input_bindings.json` and `COMPLETE.json`.
"""


# --------------------------------------------------------------------------------------------
# Frozen module (sha-checked private instance)
# --------------------------------------------------------------------------------------------

class _FrozenLoader(importlib.abc.Loader):
    def __init__(self, source, origin):
        self.source, self.origin = source, origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__dict__.setdefault('__file__', self.origin)
        exec(compile(self.source, self.origin, 'exec'), module.__dict__)


def load_frozen_rer(path=FROZEN_RER_PATH, expected_sha=FROZEN_RER_SHA256):
    """Read the frozen B(c) script once, verify its full sha256, execute those same bytes; return (module, info)."""
    with open(path, 'rb') as f:
        source = f.read()
    got = hashlib.sha256(source).hexdigest()
    if got != expected_sha:
        raise RuntimeError(f'frozen RER {path} sha256 {got} != pinned {expected_sha}')
    loader = _FrozenLoader(source, str(path))
    spec = importlib.util.spec_from_loader('r2_1_filter_rer_frozen_for_routeb', loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module, {'path': str(path), 'sha256': got, 'size': len(source)}


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------

def _arms_list():
    return [('all_tissue', '')] + [(d, b) for b in BUDGETS for d in DESIGNS]


def contrast_specs(enc, M):
    """Same order as the frozen contrast_specs() for its hoptimus block: method, design, budget."""
    specs = []
    for m in M.METHODS:
        for design in DESIGNS:
            for b in BUDGETS:
                specs.append({'encoder': enc, 'method': m, 'design': design, 'arm': design, 'budget': b,
                              'contrast_id': f'{enc}:{m}:{design}:{b}-vs-all_tissue'})
    return specs


def nonreturn_from_census(features_qf_root, read=None):
    """EXPECTED_NONRETURN per canonical budget from `_filter_summary.csv` (own_fold_query == 1 & written == 0).

    Returns (sets, bindings). `read(path) -> bytes` lets the caller hash the exact bytes parsed.
    """
    read = read or (lambda p: open(p, 'rb').read())
    sets, bindings = {}, {}
    for b in BUDGETS:
        ids = set()
        for F in range(N_FOLDS):
            p = os.path.join(features_qf_root, BUDGET_DIR[b], f'fold{F}', '_filter_summary.csv')
            data = read(p)
            bindings[f'filter_summary:{BUDGET_DIR[b]}:fold{F}'] = {
                'path': p, 'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
            df = pd.read_csv(io.BytesIO(data))
            need = {'slide_id', 'own_fold_query', 'written', 'head_fold', 'budget_percent'}
            if not need <= set(df.columns):
                raise ValueError(f'{p}: missing columns {sorted(need - set(df.columns))}')
            if set(df.head_fold.astype(int)) != {F}:
                raise ValueError(f'{p}: head_fold values {sorted(set(df.head_fold))} != {{{F}}}')
            if set(df.budget_percent.astype(float)) != {float(b)}:
                raise ValueError(f'{p}: budget_percent {sorted(set(df.budget_percent))} != {{{b}}}')
            sel = (df.own_fold_query.astype(int) == 1) & (df.written.astype(int) == 0)
            ids |= set(df.loc[sel, 'slide_id'].astype(int))
        sets[b] = ids
    return sets, bindings


def expected_nonreturn_table(enc, by_budget):
    out = {(enc, 'all_tissue', ''): set()}
    for b in BUDGETS:
        for d in DESIGNS:
            out[(enc, d, b)] = set(by_budget[b])
    return out


def routeb_config(M, root, prosper_out, pins):
    pkg = os.path.join(root, 'evaluation_qf')
    enr = {b: set(int(x) for x in pins['expected_nonreturn'][b]) for b in BUDGETS}
    return {
        'mode': 'run', 'encoder': 'uni2', 'source_run': SOURCE_RUN,
        'paths': {'population': os.path.join(prosper_out, 'r2_1_mosaic_cpu_20261003', 'attempt_001',
                                             'retrieval_population.csv'),
                  'ranked_lists': os.path.join(pkg, 'ranked_lists.csv.gz'),
                  'query_metrics': os.path.join(pkg, 'query_metrics.csv.gz'),
                  'patient_draws': os.path.join(pkg, 'patient_draws.npy'),
                  'summary': os.path.join(pkg, 'summary.json'),
                  'vote_module': None},
        'pins': {'population': POPULATION_SHA256, 'patient_draws': DRAWS_SHA256,
                 'vote_module': M.PIN['vote_module'], **{k: pins[k] for k in PIN_KEYS}},
        'expected_nonreturn': expected_nonreturn_table('uni2', enr),
        'census_root': os.path.join(root, 'features_qf'),
        'census_pins': {k: v for k, v in pins.items() if k.startswith('filter_summary:')},
        'method_interval_label': ROUTEB_METHOD_INTERVAL_LABEL,
        'expected_contrast_rows': 480,
        'tie_block_matrix_resolution': True,
        'root': root,
        'arm_bindings': os.path.join(pkg, 'arm_bindings.json'),
    }


def r0_config(M, prosper_out):
    pkg = os.path.join(prosper_out, 'r2_1_filter_20261003', 'evaluation_qf')
    exp = {('hoptimus', a, b): set(M.EXPECTED_NONRETURN[('hoptimus', a, b)]) for a, b in _arms_list()}
    return {
        'mode': 'r0', 'encoder': 'hoptimus', 'source_run': R0_SOURCE_RUN,
        'paths': {'population': os.path.join(prosper_out, 'r2_1_mosaic_cpu_20261003', 'attempt_001',
                                             'retrieval_population.csv'),
                  'ranked_lists': os.path.join(pkg, 'ranked_lists.csv.gz'),
                  'query_metrics': os.path.join(pkg, 'query_metrics.csv.gz'),
                  'patient_draws': os.path.join(pkg, 'patient_draws.npy'),
                  'summary': os.path.join(pkg, 'summary.json'),
                  'vote_module': None},
        'pins': {'population': M.PIN['population'], 'patient_draws': M.PIN['patient_draws:hoptimus'],
                 'vote_module': M.PIN['vote_module'], 'ranked_lists': M.PIN['ranked_lists:hoptimus'],
                 'query_metrics': M.PIN['query_metrics:hoptimus'], 'summary': M.PIN['summary:hoptimus']},
        'expected_nonreturn': exp,
        'census_root': os.path.join(prosper_out, 'r2_1_filter_20261003', 'features_qf'),
        'census_pins': None,
        'method_interval_label': 'descriptive (item 8c, H-Optimus)',
        'expected_contrast_rows': 480,
        'tie_block_matrix_resolution': False,  # AMENDMENT_002 rule 4: never activates in the R-0 configuration
        'run001_dir': os.path.join(prosper_out, 'r2_1_filter_rer_20261012', 'run_001'),
    }


def configure_frozen(M, cfg):
    """Install this driver's own constants as globals of the private frozen instance (read by load_arms)."""
    enc = cfg['encoder']
    M.ARMS = {enc: _arms_list()}
    M.RAW_BUDGETS = {enc: {'5', '100'}}
    M.EXPECTED_NONRETURN = dict(cfg['expected_nonreturn'])
    M.PIN = {f'{k}:{enc}' if k in ('ranked_lists', 'query_metrics', 'summary', 'patient_draws') else k: v
             for k, v in cfg['pins'].items()}
    M.PACKAGE_DIRS = {enc: (os.path.dirname(cfg['paths']['ranked_lists']),)}


# --------------------------------------------------------------------------------------------
# Amendment 002: tie-block containment evidence from the frozen no-same-patient matrix row
# --------------------------------------------------------------------------------------------

def matrix_block_counts(row, d5):
    """(n_lt, n_eq, n_library) over finite non-sentinel entries (!= 1e6, not NaN/inf) of one matrix row."""
    row = np.asarray(row, dtype=np.float64)
    lib = np.isfinite(row) & (row != SENTINEL)
    v = row[lib]
    return int(np.count_nonzero(v < d5)), int(np.count_nonzero(v == d5)), int(v.size)


def _same_bits(x, y):
    """Bitwise float64 equality of two equally shaped arrays/scalars."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
    return x.shape == y.shape and bool(np.array_equal(x.view(np.uint64), y.view(np.uint64)))


def block_reaches_stored_depth(distances, k):
    """The exact frozen `reverse_boundary_block` branch that raises 'reaches the stored depth' -> (lo, hi) or None."""
    d = np.asarray(distances, float)
    if not (d.size > k and d[k - 1] == d[k]):
        return None
    tied = np.flatnonzero(d == d[k - 1])
    lo, hi = int(tied[0]), int(tied[-1])
    if not np.array_equal(tied, np.arange(lo, hi + 1)) or hi < d.size - 1:
        return None
    return lo, hi


class TieBlockResolver:
    """Replaces `reverse_boundary_block` on the PRIVATE frozen instance (the frozen function is called first and its
    result returned unchanged). Only when the frozen rule raises because a k=5 boundary block reaches the stored depth,
    and only when enabled (route-(b) run mode), the frozen no-same-patient matrix row of the (arm, method, budget,
    query) being voted is read; containment holds iff n_lt == lo and n_eq == hi - lo + 1 (AMENDMENT_002 rule 1).

    The (arm, method, budget) and the population index of the query are supplied by `context()`, which the driver's
    own prediction loop opens around each reversed `M.arm_predictions` call; the query is identified by the identity of
    the stored distance array that the frozen `arm_predictions` passes from `arm['lists'][i]`."""

    def __init__(self, M, enabled, pop, rl, summary, state, root=None, arm_bindings_path=None):
        self.M = M
        self.enabled = bool(enabled)
        if not hasattr(M, '_routeb_frozen_reverse_boundary_block'):
            M._routeb_frozen_reverse_boundary_block = M.reverse_boundary_block
        self.frozen = M._routeb_frozen_reverse_boundary_block
        self.pop = pop
        self.rl = rl
        self.summary = summary
        self.root = root
        self.arm_bindings_path = arm_bindings_path
        self.state = state
        self.bindings = {}
        state['tie_block_matrices'] = self.bindings
        self.ctx = None
        self.activated = {}
        self._matrices = {}
        if 'reference_index' in pop.columns:
            ref = pop.reference_index.astype(int).to_numpy()
            if sorted(ref.tolist()) != list(range(len(pop))):
                raise M.GateError('tie_block_matrix_binding', 'population reference_index is not 0..n-1')
            self.col_order = np.argsort(ref, kind='mergesort')
        else:
            self.col_order = np.arange(len(pop))
        self.pop_ids_ref = [str(int(s)) for s in pop.slide_id.to_numpy()[self.col_order]]
        self.pat_ref = pop.patient_id.astype(str).to_numpy()[self.col_order]

    def install(self):
        self.M.reverse_boundary_block = self.reverse_boundary_block
        return self

    def uninstall(self):
        self.M.reverse_boundary_block = self.frozen

    class _Ctx:
        def __init__(self, owner, value):
            self.owner, self.value = owner, value

        def __enter__(self):
            self.owner.ctx = self.value

        def __exit__(self, *exc):
            self.owner.ctx = None
            return False

    def context(self, key, arm, rule):
        by_id = {id(lst[1]): i for i, lst in arm['lists'].items()}
        return self._Ctx(self, {'key': key, 'arm': arm, 'rule': rule, 'by_id': by_id})

    def reverse_boundary_block(self, labels, distances, k=None):
        k = self.M.K if k is None else k
        try:
            return self.frozen(labels, distances, k)
        except self.M.GateError as ex:
            blk = block_reaches_stored_depth(distances, k)
            if not self.enabled or blk is None or ex.gate != 'tie_block_containment' or self.ctx is None:
                raise
            i = self.ctx['by_id'].get(id(distances))
            if i is None or self.ctx['arm']['lists'][i][1] is not distances:
                raise
            return self._resolve(labels, distances, k, blk, i, ex)

    # -- matrix access ------------------------------------------------------------------------------------------

    def _arm_entry(self, a, m, raw_b):
        key = f'{a}|{m}|{raw_b}'
        hits = [x for x in self.summary.get('arms', []) if x.get('key') == key]
        if len(hits) != 1:
            raise self.M.GateError('tie_block_matrix_binding', f'summary.json arms has {len(hits)} entries for {key}')
        return key, hits[0]

    def _matrix(self, a, m, b):
        GateError = self.M.GateError
        raw_b = '' if b == '' else BUDGET_DIR[b][len('budget'):]
        key, entry = self._arm_entry(a, m, raw_b)
        if key in self._matrices:
            return self._matrices[key]
        rec_path = entry.get('matrix')
        rec_sha = entry.get('matrix_sha256')
        if not rec_path or not rec_sha:
            raise GateError('tie_block_matrix_binding', f'summary.json arm {key} lacks matrix/matrix_sha256')
        if a == 'all_tissue':
            path = rec_path
        else:
            name = QF_MATRIX_NAME.format(b=raw_b, d=DESIGN_TOKEN[a], m=m)
            path = os.path.join(self.root, 'similarity_matrix_qf', name)
            if os.path.basename(rec_path) != name or not os.path.exists(rec_path) or not os.path.samefile(
                    rec_path, path):
                raise GateError('tie_block_matrix_binding', f'{key}: summary.json matrix {rec_path} is not {path}')
        bind = {'arm_key': key, 'path': path, 'summary_path': rec_path, 'summary_sha256': rec_sha}
        if self.arm_bindings_path is not None:
            try:
                with open(self.arm_bindings_path, 'rb') as f:
                    ab_bytes = f.read()
                ab = json.loads(ab_bytes)
            except (OSError, ValueError) as ex:
                raise GateError('tie_block_matrix_binding', f'{key}: arm_bindings.json {type(ex).__name__}: {ex}')
            want_b = '-' if a == 'all_tissue' else raw_b
            ab_hits = [x for x in ab.get('arms', []) if (x.get('arm'), x.get('method'), str(x.get('budget'))) ==
                       (a, m, want_b)]
            if len(ab_hits) != 1 or ab_hits[0].get('matrix_sha256') != rec_sha:
                raise GateError('tie_block_matrix_binding', f'{key}: arm_bindings.json matrix_sha256 differs from '
                                                            'summary.json', {'arm_bindings': ab_hits})
            bind.update(arm_bindings_path=self.arm_bindings_path,
                        arm_bindings_sha256=hashlib.sha256(ab_bytes).hexdigest())
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except OSError as ex:
            raise GateError('tie_block_matrix_binding', f'{key}: {type(ex).__name__}: {ex}')
        got = hashlib.sha256(data).hexdigest()
        bind.update(sha256=got, size=len(data), match=got == rec_sha)
        self.bindings[key] = bind
        if got != rec_sha:
            raise GateError('tie_block_matrix_binding', f'{key}: matrix sha256 {got} != summary.json {rec_sha}', bind)
        df = pd.read_pickle(io.BytesIO(data))  # bytes hash-verified above
        if not isinstance(df, pd.DataFrame) or df.to_numpy().dtype.kind != 'f':
            raise GateError('tie_block_matrix_binding', f'{key}: not a floating DataFrame')
        rows = [str(x) for x in df.index]
        cols = [str(c) for c in df.columns]
        if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
            raise GateError('tie_block_matrix_binding', f'{key}: duplicate axis ids')
        ci = {c: j for j, c in enumerate(cols)}
        if a != 'all_tissue' and cols != self.pop_ids_ref:
            raise GateError('tie_block_matrix_binding', f'{key}: columns are not the population in reference order')
        missing = [s for s in self.pop_ids_ref if s not in ci]
        if missing:
            raise GateError('tie_block_matrix_binding', f'{key}: {len(missing)} population slides missing from columns')
        mat = {'key': key, 'bind': bind, 'values': df.to_numpy().astype(np.float64, copy=False),
               'row_of': {r: j for j, r in enumerate(rows)},
               'col_sel': np.array([ci[s] for s in self.pop_ids_ref], dtype=np.int64)}
        self._matrices[key] = mat
        return mat

    def _stored_neighbours(self, m, a, b, qid):
        raw = self.rl.budget.map(self.M._canon_budget)
        g = self.rl[(self.rl.method == m) & (self.rl.arm == a) & (raw == b) & (self.rl['query'].astype(int) == qid)]
        g = g.sort_values('rank', kind='mergesort')
        return g['rank'].astype(int).to_numpy(), g['neighbour'].astype(int).to_numpy(), \
            g['distance'].to_numpy(dtype=np.float64)

    # -- the amended rule ---------------------------------------------------------------------------------------

    def _resolve(self, labels, distances, k, blk, i, ex):
        GateError = self.M.GateError
        e, m, a, b = self.ctx['key']
        qid = int(self.pop.slide_id.iat[i])
        lo, hi = blk
        d = np.asarray(distances, dtype=np.float64)
        n = int(d.size)
        d5 = d[k - 1]
        rk = (a, m, b, qid)
        rec = self.activated.get(rk)
        if rec is None:
            mat = self._matrix(a, m, b)
            r = mat['row_of'].get(str(qid))
            if r is None:
                raise GateError('tie_block_matrix_consistency', f"{mat['key']}: query {qid} is not a matrix row")
            row = mat['values'][r, mat['col_sel']]  # population (reference_index) column order
            lib = np.isfinite(row) & (row != SENTINEL)
            qpat = str(self.pop.patient_id.iat[i])
            if np.any(lib & (self.pat_ref == qpat)):
                raise GateError('tie_block_matrix_consistency', f"{mat['key']} query {qid}: a same-patient library "
                                                                'cell is not the 1e6 sentinel')
            n_lt, n_eq, n_lib = matrix_block_counts(row, d5)
            vals = row[lib]
            ids = np.array(self.pop_ids_ref, dtype=object)[lib]
            order = np.argsort(vals, kind='mergesort')
            top_vals = vals[order[:n]]
            ranks, nbs, rl_d = self._stored_neighbours(m, a, b, qid)
            problems = []
            if top_vals.size != n or not _same_bits(top_vals, d):
                problems.append(f'stored top-{n} distances != the matrix row {n} smallest (bitwise)')
            if not (np.array_equal(ranks, np.arange(1, n + 1)) and _same_bits(rl_d, d)):
                problems.append('ranked_lists rows for this query differ from the stored list')
            col = {s: j for j, s in enumerate(self.pop_ids_ref)}
            if len(set(nbs.tolist())) != nbs.size:
                problems.append('stored neighbour ids are not distinct')
            for rr, (nb, dd) in enumerate(zip(nbs.tolist(), d.tolist())):
                j = col.get(str(nb))
                if j is None or not lib[j] or not _same_bits(row[j], dd):
                    problems.append(f'stored neighbour {nb} at rank {rr + 1} is not a library column at '
                                    f'distance {dd!r}')
            if problems:
                raise GateError('tie_block_matrix_consistency', f"{mat['key']} query {qid}: matrix row and stored "
                                                                'list disagree', problems)
            contained = n_lt == lo and n_eq == hi - lo + 1
            rec = {'arm': a, 'method': m, 'budget': b, 'query': qid, 'stored_depth': n, 'd5': float(d5),
                   'lo': lo, 'hi': hi, 'block_ranks': [lo + 1, hi + 1], 'n_lt': n_lt, 'n_eq': n_eq,
                   'n_library': n_lib,
                   'stable_top_ids_equal_stored_order': [str(x) for x in ids[order[:n]]] == [str(x) for x in nbs],
                   'matrix': {'path': mat['bind']['path'], 'sha256': mat['bind']['sha256']},
                   'decision': 'contained' if contained else 'not_contained', 'rules': []}
            self.activated[rk] = rec
        rec['rules'].append(self.ctx['rule'])
        if rec['decision'] != 'contained':
            raise GateError(ex.gate, ex.reason, rec)
        labels = list(labels)
        labels[lo:hi + 1] = labels[lo:hi + 1][::-1]
        return labels, True

    def evidence(self, arms, mode, enc):
        totals = []
        for key in sorted(arms):
            arm = arms[key]
            reach = sum(block_reaches_stored_depth(dd, self.M.K) is not None for _, dd in arm['lists'].values())
            _, m, a, b = key
            res = sum(1 for (aa, mm, bb, _q), r in self.activated.items()
                      if (aa, mm, bb) == (a, m, b) and r['decision'] == 'contained')
            totals.append({'encoder': key[0], 'method': m, 'arm': a, 'budget': b,
                           'returned_queries': int((~arm['nonret']).sum()),
                           'boundary_tie_queries': int(arm['boundary_tie'].sum()),
                           'block_reaches_stored_depth': int(reach), 'resolved_by_matrix': int(res),
                           'stored_depths': sorted({int(len(dd)) for _, dd in arm['lists'].values()})})
        return {'amendment': AMENDMENT, 'mode': mode, 'encoder': enc, 'matrix_resolution_enabled': self.enabled,
                'rule': ('a k=5 boundary tie block reaching the last stored rank is contained iff, over finite '
                         'non-sentinel (!= 1e6, not NaN) library entries of the frozen _no_same_patient.pkl row, '
                         'n_lt = #{d < d5} == lo and n_eq = #{d == d5} == hi - lo + 1 (lo/hi 0-based stored '
                         'indices); otherwise tie_block_containment fails'),
                'n_activated': len(self.activated),
                'activated': [self.activated[k_] for k_ in sorted(self.activated)],
                'totals': totals}


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------

def method_contrasts(M, specs, rep_store, con_rows, bootstrap, label):
    """Frozen method_contrasts() arithmetic without its ENC_S5-keyed interval label (§12.16)."""
    macro = {(r['contrast_id'], r['rule'], r['handling']): r for r in con_rows if r['class'] == M.MACRO}
    by = {}
    for sp in specs:
        by.setdefault((sp['encoder'], sp['design'], sp['budget']), {})[sp['method']] = sp['contrast_id']
    rows = []
    for (e, design, b), d in by.items():
        for rule in M.RULES:
            for handling in M.HANDLINGS:
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
                    row['rer_yx_minus_ot_ci_low'], row['rer_yx_minus_ot_ci_high'] = M.percentile_ci(diff)
                    row['n_undefined_reps'] = int(np.sum(~np.isfinite(diff)))
                    dd = py['dM'] - po['dM']
                    row['delta_f1_yx_minus_ot_ci_low'], row['delta_f1_yx_minus_ot_ci_high'] = M.percentile_ci(dd)
                else:
                    row.update(rer_yx_minus_ot_ci_low=np.nan, rer_yx_minus_ot_ci_high=np.nan, n_undefined_reps=np.nan,
                               delta_f1_yx_minus_ot_ci_low=np.nan, delta_f1_yx_minus_ot_ci_high=np.nan)
                row['interval_label'] = label
                row['quantile_method'] = M.QUANTILE_METHOD
                rows.append(row)
    return rows


def run_pipeline(M, cfg, out, bootstrap=True, sizes=None, vinfo_frozen=None):
    """Execute one configuration into `out`. Raises M.GateError on any gate failure. Returns a summary dict.

    `sizes` (tests only) overrides (n_queries, n_patients, n_groups, n_reps); production uses the frozen constants.
    """
    GateError = M.GateError
    n_queries, n_patients, n_groups, n_reps = sizes or (M.N_QUERIES, M.N_PATIENTS, M.N_GROUPS, M.N_REPS)
    enc = cfg['encoder']
    configure_frozen(M, cfg)
    gates = {}
    paths = dict(cfg['paths'])
    if paths.get('vote_module') is None:
        paths['vote_module'] = os.path.join(cfg['repo'], 'scripts', 'r2_3_internal_failure_tables.py')
    bindings, bad, data = M.hash_inputs(paths, pins=cfg['pins'])
    state = {'inputs': bindings, 'frozen_rer': vinfo_frozen, 'driver': driver_identity(), 'mode': cfg['mode'],
             'encoder': enc, 'source_run': cfg['source_run']}
    run_pipeline.state = state
    if bad:
        raise GateError('input_hashes', f'{len(bad)} input(s) missing or differ from the pinned sha256', bad)
    gates['input_hashes'] = f'{len(bindings)} inputs match pins'

    plurality, legacy_generous, vinfo = M.load_vote_functions(paths['vote_module'], expected_sha=cfg['pins'][
        'vote_module'], source=data['vote_module'])
    gates['vote_module'] = 'pinned bytes executed via spec_from_loader'

    pop = pd.read_csv(io.BytesIO(data['population']))
    if not (len(pop) == n_queries and pop.slide_id.is_unique and pop.patient_id.nunique() == n_patients
            and pop.diagnostic_group.nunique() == n_groups):
        raise GateError('population', f'population is not {n_queries} unique slides / {n_patients} patients / '
                                      f'{n_groups} groups')
    groups = sorted(pop.diagnostic_group.unique())
    truth = pop.diagnostic_group.to_numpy()
    patients_sorted = sorted(pop.patient_id.unique())
    gates['population'] = f'{n_queries} slides / {n_patients} patients / {n_groups} groups'

    summary = json.loads(data['summary'])
    draws = np.load(io.BytesIO(data['patient_draws']), allow_pickle=False)
    M.check_draws(draws, patients_sorted, {enc: summary}, n_reps=n_reps, n_patients=n_patients)
    gates['draws_binding'] = (f'shape ({n_reps}, {n_patients}) int64, row sums {n_patients}, columns = sorted unique '
                              'patient_id = summary.json patient_draws_patients; sha256 pinned')

    # EXPECTED_NONRETURN: the pin (independent of the evaluator) must equal the in-job census re-derivation.
    census_bytes = {}

    def _read(p):
        with open(p, 'rb') as f:
            b = f.read()
        census_bytes[p] = b
        return b
    try:
        census, census_bind = nonreturn_from_census(cfg['census_root'], read=_read)
    except (OSError, ValueError) as ex:
        raise GateError('nonreturn_census', f'{type(ex).__name__}: {ex}')
    state['census'] = census_bind
    if cfg.get('census_pins') is not None:
        diff = {k: {'pinned': cfg['census_pins'].get(k), 'observed': v['sha256']} for k, v in census_bind.items()
                if cfg['census_pins'].get(k) != v['sha256']}
        if diff or set(cfg['census_pins']) != set(census_bind):
            raise GateError('nonreturn_census', '_filter_summary.csv bytes differ from the pin file', diff)
    census_tab = expected_nonreturn_table(enc, census)
    if {k: set(v) for k, v in census_tab.items()} != {k: set(v) for k, v in cfg['expected_nonreturn'].items()}:
        raise GateError('nonreturn_census', 'EXPECTED_NONRETURN pin != _filter_summary.csv census',
                        {str(k): {'pin': sorted(cfg['expected_nonreturn'].get(k, ())), 'census': sorted(v)}
                         for k, v in census_tab.items()})
    gates['nonreturn_census'] = 'EXPECTED_NONRETURN pin == _filter_summary.csv (own_fold_query==1 & written==0)'

    rl = pd.read_csv(io.BytesIO(data['ranked_lists']), compression='gzip', dtype={'budget': str})
    qm = pd.read_csv(io.BytesIO(data['query_metrics']), compression='gzip', dtype={'budget': str})
    arms = {(enc,) + key: a for key, a in M.load_arms(enc, rl, qm, pop).items()}
    observed = {k: a['nonret_ids'] for k, a in arms.items()}
    expected = {(e, m, a, b): cfg['expected_nonreturn'][(e, a, b)] for (e, m, a, b) in arms}
    M.check_nonreturn_sets(observed, expected)
    gates['query_status'] = f'{len(arms)} arms x {n_queries} k={M.K} rows; status in {list(M.ALLOWED_STATUS)}'
    gates['nonreturn_sets'] = 'evaluator non-returned sets == EXPECTED_NONRETURN for every arm/method/budget'
    gates['ranked_list_integrity'] = 'frozen load_arms checks (ranks, population neighbours, groups, relevance, ' \
                                     'no same-patient neighbour)'
    gates['hit5_cross_check'] = 'hit@5 from ranked_lists == conditional_hit for every returned query'
    gates['boundary_tie_agreement'] = 'd5 == d6 detection == query_metrics boundary_tie'

    resolver = TieBlockResolver(M, cfg.get('tie_block_matrix_resolution', False), pop, rl, summary, state,
                                root=cfg.get('root'), arm_bindings_path=cfg.get('arm_bindings')).install()
    preds, preds_rev = {}, {}
    try:
        for key, arm in arms.items():
            for rule in M.RULES:
                preds[key + (rule,)] = M.arm_predictions(arm, truth, rule, plurality, legacy_generous)
                with resolver.context(key, arm, rule):
                    preds_rev[key + (rule,)] = M.arm_predictions(arm, truth, rule, plurality, legacy_generous, True)
    finally:
        resolver.uninstall()
    tie_evidence = resolver.evidence(arms, cfg['mode'], enc)
    n_reach = sum(t['block_reaches_stored_depth'] for t in tie_evidence['totals'])
    n_res = sum(t['resolved_by_matrix'] for t in tie_evidence['totals'])
    if n_reach != n_res or any(r['decision'] != 'contained' or sorted(r['rules']) != sorted(M.RULES)
                               for r in tie_evidence['activated']):
        raise GateError('tie_block_containment', f'{n_reach} block(s) reach the stored depth but {n_res} were '
                                                 'resolved by the matrix row', tie_evidence)
    if n_res:
        gates['tie_block_containment'] = ('every k=5 boundary tie block lies within the stored top-10, or (amendment '
                                          f'002, {n_res} list row(s)) ends exactly at the stored depth per the frozen '
                                          'no-same-patient matrix row: n_lt == lo and n_eq == hi - lo + 1')
        gates['tie_block_matrix_evidence'] = ('matrix sha256 == summary.json arms (and arm_bindings.json); stored '
                                              'distances == matrix row smallest (bitwise); stored neighbours are '
                                              'library columns at those distances; same-patient cells are 1e6')
    else:
        gates['tie_block_containment'] = 'every k=5 boundary tie block lies within the stored top-10'

    W = None
    if bootstrap:
        pcol = {p: i for i, p in enumerate(patients_sorted)}
        qcol = np.array([pcol[p] for p in pop.patient_id])
        W = draws[:, qcol].astype(np.float64)
    gi = {g: i for i, g in enumerate(groups)}
    T = np.zeros((len(pop), len(groups)))
    T[np.arange(len(pop)), [gi[t] for t in truth]] = 1.0
    PM = {k: M.prediction_matrix(v, groups) for k, v in preds.items()}
    PM_rev = {k: M.prediction_matrix(v, groups) for k, v in preds_rev.items()}

    est_rows, con_rows, pq_rows, tie_rows = [], [], [], []
    rep_store = {}
    specs = contrast_specs(enc, M)
    for sp in specs:
        e, m = sp['encoder'], sp['method']
        kA = (e, m, 'all_tissue', '')
        kF = (e, m, sp['arm'], sp['budget'])
        armA, armF = arms[kA], arms[kF]
        for rule in M.RULES:
            for handling in M.HANDLINGS:
                inc = M.inclusion_mask(handling, armA['nonret'], armF['nonret'])
                res = M.evaluate_contrast(W, T, PM[kA + (rule,)], PM[kF + (rule,)], inc)
                rows, reps = M.summarise_contrast(res, groups, n_reps)
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
                            'n_zero_support_reps': r['n_zero_support_reps'], 'quantile_method': M.QUANTILE_METHOD})
                    for i in np.flatnonzero(incb):
                        pq_rows.append((int(pop.slide_id.iat[i]), truth[i], e, m, arm['arm'], sp['design'],
                                        arm['budget'], rule, handling, sp['contrast_id'], arm['status'][i],
                                        int(arm['boundary_tie'][i]), preds[key + (rule,)][i]))
                for r in rows:
                    # Route (b) / H-Optimus-1: no submitted counterpart; no ENC_S5-keyed path is called (§12.16).
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
                        'submitted_delta': np.nan,
                        'submitted_population_note': '',
                        'interval_label': ('prespecified_item8 (unadjusted)'
                                           if r['class'] in M.PRESPECIFIED_GROUPS + (M.MACRO,)
                                           else 'unadjusted, descriptive'),
                        'quantile_method': M.QUANTILE_METHOD})
                if m == 'yottixel':
                    tie_rows.extend(M.tie_sensitivity_rows(sp, rule, handling, inc, T, groups,
                                                           PM, PM_rev, kA, kF, rows))
    gates['point_support'] = 'every class has positive support at the point estimate in every contrast'

    method_rows = method_contrasts(M, specs, rep_store, con_rows, bootstrap, cfg['method_interval_label'])
    est = pd.DataFrame(est_rows, columns=ESTIMATE_COLUMNS)
    con = pd.DataFrame(con_rows, columns=CONTRAST_COLUMNS)
    check_contrast_schema(con, enc, n_groups, cfg['expected_contrast_rows'] if sizes is None else None)
    gates['output_schema'] = (f"contrasts.csv: run_001's 27 columns in order; {len(con)} rows; submitted_delta NaN "
                              "and submitted_population_note '' for every row; contrast_id "
                              f"{enc}:{{method}}:{{design}}:{{budget}}-vs-all_tissue")
    mc = pd.DataFrame(method_rows)
    ts = pd.DataFrame(tie_rows)
    tc = pd.DataFrame(M.tie_count_rows(arms, preds, preds_rev))
    pq = pd.DataFrame(pq_rows, columns=PQ_COLUMNS)
    outputs = {
        'estimates.csv': M._csv_bytes(est), 'contrasts.csv': M._csv_bytes(con),
        'method_contrasts.csv': M._csv_bytes(mc), 'tie_sensitivity.csv': M._csv_bytes(ts),
        'tie_counts.csv': M._csv_bytes(tc),
        'per_query_predictions.csv.gz': gzip.compress(M._csv_bytes(pq), mtime=0),
        'reconciliation.json': M._json_bytes(NOT_APPLICABLE),
        'lineage_check.json': M._json_bytes(NOT_APPLICABLE),
        RESOLUTION_FILE: M._json_bytes(tie_evidence),
    }
    if cfg['mode'] == 'run':
        outputs['README.md'] = README_TEXT.encode()
    state['versions'] = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}
    state['vote_module_exec'] = vinfo
    outputs['input_bindings.json'] = M._json_bytes(state)
    return {'gates': gates, 'outputs': outputs, 'n_contrasts': len(specs), 'n_reps': n_reps,
            'counts': {'estimates': len(est), 'contrasts': len(con), 'method_contrasts': len(mc),
                       'tie_sensitivity': len(ts), 'tie_counts': len(tc), 'per_query': len(pq)}}


def check_contrast_schema(con, enc, n_groups, expected_rows):
    """§3 A-5 / review 001 M2: exact columns, NaN submitted_delta, '' note, contrast_id form, row count."""
    if list(con.columns) != CONTRAST_COLUMNS:
        raise ValueError(f'contrasts columns {list(con.columns)} != run_001 27 columns')
    if not con['submitted_delta'].isna().all():
        raise ValueError('submitted_delta is not NaN for every row')
    if not (con['submitted_population_note'] == '').all():
        raise ValueError("submitted_population_note is not '' for every row")
    if set(con.encoder) != {enc} or not set(con.design) <= set(DESIGNS) or not (con.arm == con.design).all():
        raise ValueError('encoder/design/arm values outside the route-(b) set')
    if not set(con.budget) <= set(BUDGETS):
        raise ValueError(f'budget values {sorted(set(con.budget))} outside {BUDGETS}')
    want = enc + ':' + con.method + ':' + con.design + ':' + con.budget + '-vs-all_tissue'
    if not (con.contrast_id == want).all():
        raise ValueError('contrast_id differs from {enc}:{method}:{design}:{budget}-vs-all_tissue')
    if expected_rows is not None and len(con) != expected_rows:
        raise ValueError(f'{len(con)} contrast rows != {expected_rows}')
    if expected_rows is not None and len(con) != 8 * 2 * 2 * (n_groups + 1):
        raise ValueError('row count is not 8 contrasts x 2 rules x 2 handlings x (groups + macro)')
    return True


# --------------------------------------------------------------------------------------------
# R-0 comparison
# --------------------------------------------------------------------------------------------

def _select_rows_bytes(M, data, encoder):
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    return M._csv_bytes(df[df.encoder == encoder])


def _rewrite_bytes(M, data):
    return M._csv_bytes(pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False))


def r0_compare(M, run001_dir, outputs, encoder='hoptimus', pins=None, diagnostic_pins=None):
    """Specification R-0: run_001 `encoder` rows of contrasts/estimates (selected, re-written with the same CSV writer)
    must equal this run's files byte-identically. Diagnostics for the other CSVs are reported, not gated."""
    pins = pins or {'contrasts.csv': RUN001_CONTRASTS_SHA256, 'estimates.csv': RUN001_ESTIMATES_SHA256}
    diagnostic_pins = RUN001_DIAGNOSTIC_SHA256 if diagnostic_pins is None else diagnostic_pins
    res = {'gate_files': {}, 'diagnostic_files': {}}
    ok = True
    for name, pin in pins.items():
        with open(os.path.join(run001_dir, name), 'rb') as f:
            ref = f.read()
        ref_sha = hashlib.sha256(ref).hexdigest()
        sel = _select_rows_bytes(M, ref, encoder)
        mine = outputs[name]
        rec = {'run001_sha256': ref_sha, 'run001_pin_match': ref_sha == pin,
               'selected_sha256': hashlib.sha256(sel).hexdigest(), 'r0_sha256': hashlib.sha256(mine).hexdigest(),
               'r0_rewritten_sha256': hashlib.sha256(_rewrite_bytes(M, mine)).hexdigest()}
        rec['byte_identical'] = sel == mine
        rec['byte_identical_after_same_writer'] = sel == _rewrite_bytes(M, mine)
        rec['pass'] = rec['run001_pin_match'] and rec['byte_identical'] and rec['byte_identical_after_same_writer']
        ok &= rec['pass']
        res['gate_files'][name] = rec
    for name, pin in diagnostic_pins.items():
        p = os.path.join(run001_dir, name)
        try:
            with open(p, 'rb') as f:
                ref = f.read()
            sel = _select_rows_bytes(M, ref, encoder)
            res['diagnostic_files'][name] = {'run001_pin_match': hashlib.sha256(ref).hexdigest() == pin,
                                             'byte_identical': sel == outputs[name]}
        except Exception as ex:  # diagnostics never gate
            res['diagnostic_files'][name] = {'error': f'{type(ex).__name__}: {ex}'}
    res['pass'] = bool(ok)
    return res


# --------------------------------------------------------------------------------------------
# Pins (derive-pins)
# --------------------------------------------------------------------------------------------

def derive_pins(root):
    """Route-(b) pin file content from the census and evaluation_qf/summary.json (small files only)."""
    census, bind = nonreturn_from_census(os.path.join(root, 'features_qf'))
    spath = os.path.join(root, 'evaluation_qf', 'summary.json')
    with open(spath, 'rb') as f:
        sbytes = f.read()
    s = json.loads(sbytes)
    outs = s.get('outputs', {})
    pins = {'summary': hashlib.sha256(sbytes).hexdigest(),
            'ranked_lists': outs['ranked_lists.csv.gz']['sha256'],
            'query_metrics': outs['query_metrics.csv.gz']['sha256']}
    if outs.get('patient_draws.npy', {}).get('sha256') != DRAWS_SHA256:
        raise ValueError(f"evaluation_qf patient_draws.npy sha {outs.get('patient_draws.npy')} != {DRAWS_SHA256}")
    if s.get('population', {}).get('sha256') != POPULATION_SHA256:
        raise ValueError('evaluation_qf summary population sha differs from the contract pin')
    pins.update({k: v['sha256'] for k, v in bind.items()})
    pins['expected_nonreturn'] = {b: sorted(census[b]) for b in BUDGETS}
    pins['derivation'] = ('expected_nonreturn: _filter_summary.csv own_fold_query == 1 & written == 0, union over '
                          'folds 0-4 per budget, identical for query_only and both_filtered; all_tissue = empty. '
                          'ranked_lists/query_metrics: evaluation_qf summary.json outputs; summary: file sha256.')
    return pins


def load_pins(path, expected_sha):
    with open(path, 'rb') as f:
        b = f.read()
    got = hashlib.sha256(b).hexdigest()
    if got != expected_sha:
        raise ValueError(f'pin file {path} sha256 {got} != --pins-sha256 {expected_sha}')
    pins = json.loads(b)
    missing = [k for k in PIN_KEYS + ('expected_nonreturn',) if k not in pins]
    if missing or set(pins['expected_nonreturn']) != set(BUDGETS):
        raise ValueError(f'pin file lacks {missing or "expected_nonreturn budgets"}')
    return pins, {'path': path, 'sha256': got, 'size': len(b)}


# --------------------------------------------------------------------------------------------
# Output / main
# --------------------------------------------------------------------------------------------

def driver_identity():
    p = os.path.abspath(__file__)
    with open(p, 'rb') as f:
        b = f.read()
    return {'path': p, 'sha256': hashlib.sha256(b).hexdigest(), 'size': len(b)}


def _write_failed(M, out, gate, reason, detail, state):
    obj = {'status': 'FAILED', 'gate': gate, 'reason': reason, 'detail': detail,
           'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'lsf_job': os.environ.get('LSB_JOBID'), 'host': socket.gethostname(), 'input_bindings': state}
    M._write(out, 'FAILED.json', M._json_bytes(obj))
    M.write_sha256sums(out)


def _finish(M, out, summary, cfg, extra):
    for name, data in summary['outputs'].items():
        M._write(out, name, data)
    complete = {
        'status': 'COMPLETE', 'contract': 'R2.1-ROUTE-B-UNI2 candidate 002 (accepted with LEAD_RESOLUTIONS_002 + '
                                          'ADDENDUM); B(c) RER per ADR0021 candidate 002',
        'source_run': cfg['source_run'], 'amendment': AMENDMENT, 'encoder': cfg['encoder'], 'mode': cfg['mode'],
        'bootstrap': extra['bootstrap'], 'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'lsf_job': os.environ.get('LSB_JOBID'), 'lsf_queue': os.environ.get('LSB_QUEUE'),
        'host': socket.gethostname(), 'platform': platform.platform(), 'argv': sys.argv,
        'gates': summary['gates'], 'n_contrasts': summary['n_contrasts'], 'counts': summary['counts'],
        'undefined_threshold': M.UNDEFINED_THRESHOLD, 'max_undefined_reps_per_class': M.max_undefined_reps(
            summary['n_reps']),
        'code': {'driver': driver_identity(), 'frozen_rer': extra['frozen']},
        'outputs': {n: M.sha256_bytes(b) for n, b in sorted(summary['outputs'].items())},
        'notes': ['route identity: directory, this source_run field and contrast_id; no source_run column',
                  'submitted_delta NaN / submitted_population_note empty for every row (no submitted counterpart)',
                  'legacy_generous_ORACLE is an oracle rule (uses the truth label to break ties)',
                  'all intervals are unadjusted, descriptive']}
    if 'r0' in extra:
        complete['r0'] = extra['r0']
    M._write(out, 'COMPLETE.json', M._json_bytes(complete))
    M.write_sha256sums(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='mode', required=True)
    for name in ('run', 'r0'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True, help='output directory (must not exist or be empty)')
        p.add_argument('--repo', default=REPO_DEFAULT)
        p.add_argument('--root', default=None, help='package ROOT (default PROSPER_OUT/' + ROOT_NAME + ')')
        p.add_argument('--no-bootstrap', action='store_true', help='tests only: skip the bootstrap intervals')
        if name == 'run':
            p.add_argument('--pins', required=True, help='route-(b) pin file written by derive-pins')
            p.add_argument('--pins-sha256', required=True, help='full sha256 of the pin file (custody)')
    p = sub.add_parser('derive-pins')
    p.add_argument('--root', default=None)
    p.add_argument('--out-file', required=True)
    args = ap.parse_args(argv)

    prosper_out = os.environ.get('PROSPER_OUT')
    if not prosper_out:
        print('error: PROSPER_OUT must be exported', file=sys.stderr)
        return 2
    root = os.path.abspath(args.root or os.path.join(prosper_out, ROOT_NAME))

    if args.mode == 'derive-pins':
        if os.path.exists(args.out_file):
            print(f'error: {args.out_file} exists', file=sys.stderr)
            return 2
        pins = derive_pins(root)
        with open(args.out_file, 'x') as f:
            json.dump(pins, f, indent=1, sort_keys=True)
            f.write('\n')
        print(hashlib.sha256(open(args.out_file, 'rb').read()).hexdigest(), args.out_file)
        return 0

    out = os.path.abspath(args.out)
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        print(f'error: --out {out} exists and is not an empty directory', file=sys.stderr)
        return 2
    want_parent = os.path.join(root, 'fidelity', 'r0') if args.mode == 'r0' else root
    if os.path.dirname(out) != want_parent:
        print(f'error: --out must be a child of {want_parent}', file=sys.stderr)
        return 2
    M, finfo = load_frozen_rer()
    os.makedirs(out, exist_ok=True)
    run_pipeline.state = None
    cfg = None
    try:
        if args.mode == 'run':
            pins, pinfo = load_pins(args.pins, args.pins_sha256)
            cfg = routeb_config(M, root, prosper_out, pins)
        else:
            cfg = r0_config(M, prosper_out)
            pinfo = None
        cfg['repo'] = os.path.abspath(args.repo)
        summary = run_pipeline(M, cfg, out, bootstrap=not args.no_bootstrap, vinfo_frozen=finfo)
        extra = {'bootstrap': not args.no_bootstrap, 'frozen': finfo}
        if pinfo is not None:
            summary['gates']['pin_file'] = f"{pinfo['path']} sha256 {pinfo['sha256']}"
        if args.mode == 'r0':
            cmp_ = r0_compare(M, cfg['run001_dir'], summary['outputs'])
            summary['outputs']['r0_compare.json'] = M._json_bytes(cmp_)
            extra['r0'] = {'pass': cmp_['pass']}
            if not cmp_['pass']:
                for name, data in summary['outputs'].items():
                    M._write(out, name, data)
                raise M.GateError('r0', 'R-0 rows differ from run_001 hoptimus rows', cmp_['gate_files'])
            summary['gates']['r0'] = 'contrasts.csv and estimates.csv == run_001 hoptimus rows (byte-identical)'
        _finish(M, out, summary, cfg, extra)
    except M.GateError as ex:
        _write_failed(M, out, ex.gate, ex.reason, ex.detail, getattr(run_pipeline, 'state', None))
        print(f'FAILED gate={ex.gate}: {ex.reason} (see {out}/FAILED.json)')
        return 1
    except Exception as ex:  # never a silent pass
        _write_failed(M, out, 'unexpected_error', f'{type(ex).__name__}: {ex}', traceback.format_exc(),
                      getattr(run_pipeline, 'state', None))
        print(f'FAILED unexpected {type(ex).__name__}: {ex} (see {out}/FAILED.json)')
        return 1
    print(f"COMPLETE {cfg['mode']}: {len(summary['gates'])} gates; {summary['counts']} -> {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
