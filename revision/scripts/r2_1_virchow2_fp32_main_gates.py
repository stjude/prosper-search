#!/usr/bin/env python3
"""V2-main M3.5 completeness/structure gates, M4.1 ranks resolution, A9 host census and package COMPLETE/FAILED.

Specification: OUT/r2_1_virchow2_fp32_main_20261013/contract_candidate_002/SPECIFICATION.md (+ LEAD_RESOLUTIONS_002 and its
ADDENDUM A1-A10). Subcommands:

  m3        M3.5.1-M3.5.5 over the run stage (row files, stderr census -> stderr_census.tsv, provenance counts, 8
            matrices, self-sentinel rule, provenance fields, descriptive diffs), A9 host_census.tsv,
            run_resources.json; writes <run>/m3_matrix_gates.json, and <run>/FAILED on any failure.
            AMENDMENT_004: M3.5.4 aggregate `feature_dim` is checked per method against AGG_FEATURE_DIM
            (median_pooling 1,280 = WIDTH; sampler 12,800 = 10 x WIDTH; `feature_dim_matches_method`, replacing
            `feature_dim_1280`) and against <fidelity-dir>/m1_rebuild_<m>.json provenance.feature_dim
            (`fidelity_rebuild_feature_dim_consistent`). For a run dir holding LINEAGE_RUN_002.json (required when
            the run dir is named run_002) it also records amendment_004 / run_001_failed / lineage_file bindings;
            `--amendment` (default F.AMENDMENT_004) must hash to F.AMENDMENT_004_SHA and to the lineage record.
  m4-ranks  M4.1: the resolved matrix path of each metrics_cli run -> <run>/m4_ranks_resolution.json.
            ADDENDUM_003: the path is read from `config.input_similarity_matrix` (accepted metrics_cli shape), else
            the top-level key; `input_similarity_matrix_key` ("config"|"top_level"|null) and the full `config` block
            are recorded. Outputs carried into run_003 by lineage name the run_002 copies; such a path is accepted
            only via the stage's own lineage record (same relpath in the lineage source, listed, hash-identical;
            `matrix_resolution`/`ranked_lists_resolution` route "lineage"). The report records the gates/fidelity
            script hashes (`script_sha256`) and each step's launch.sha256/step.json binding.
  tie-hosts A9: host diversity among order-sensitive Virchow2 fp32 cells of the Run A tie census.
  input-bindings  M0.3: re-hash the §I inputs a stage consumes -> <stage>/input_bindings.json (STOP on mismatch).
  package   M6 COMPLETE rule [Q2]: per-directory row manifests, SHA256SUMS, then exactly one of COMPLETE.json
            (Run B COMPLETE and A3 passed), PARTIAL.json (status COMPLETE_RUN_A_ONLY, replacement_products pending;
            Run B infeasible/not released, LEAD_IMPL_DECISIONS_001 F4) or FAILED. The compare report must agree with
            Run B existence, A3 and the specification bootstrap mode (F1/F15).
  --run-dir defaults to run_003 (ADDENDUM_003; run_001 and run_002 are frozen FAILED). ADDENDUM_002 F4: m3/m4-ranks/
  tie-hosts refuse (writing nothing) a --run-dir whose basename is not RUN_STAGE (run_003) unless
  --allow-run-dir-mismatch, and
  any run dir already holding FAILED; the resolved run dir and RUN_STAGE are recorded in their reports
  (`run_dir_guard`). ADDENDUM_002 F8: a malformed fidelity m1_rebuild_<m>.json (invalid JSON, non-object, non-object
  `provenance`) is a recorded M1.2 gate failure (m3_matrix_gates.json + FAILED), never a bare exit.
  --fidelity-dir defaults to F.FIDELITY_DIR (fidelity_002 under AMENDMENT_003; fidelity_001 is frozen FAILED); the
  package entry `fidelity_COMPLETE` records the fidelity directory actually used.

Shared helpers come from the hash-checked fidelity script of this package (loaded by path). No LSF calls; nothing is
ever overwritten (a FAILED is final; a new attempt uses a new stage: run_003/ under ADDENDUM_003).
"""
import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('v2main_fidelity_helpers_gates',
                                               os.path.join(HERE, 'r2_1_virchow2_fp32_main_fidelity.py'))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

ROOT = F.ROOT
RUN_DIR = F.RUN_DIR_003            # ADDENDUM_003 stage (init-run lineage from frozen FAILED run_002)
RUN_STAGE = os.path.basename(F.RUN_DIR_003)     # 'run_003'; mirrors submit.py RUN_STAGE (ADDENDUM_002 F4)
CONTRACT = os.path.join(ROOT, 'contract_candidate_002', 'CONTRACT.md')
CONTRACT_SHA = 'f0ef449ff2bd8708e07afa5a1e68a94d71e02ec3e057b2c6057e40fe7a98cca2'
MODEL = 'virchow2_fp32'
N_OT_ELEMENTS = None          # = N_SLIDES (one element per row); resolved at call time so tests can shrink N
BOB_BATCH = 20
JOBS = {'ot': ('v2m_m3_ot_canary', 'v2m_m3_ot_fanout'), 'bob': ('v2m_m3_bob_canary', 'v2m_m3_bob_fanout')}
AGG_METHODS = ('median_pooling', 'sampler')
# AMENDMENT_004: slide-vector width the frozen stock aggregate_cli records as feature_dim per aggregate method
# (sampler = 10 percentiles x WIDTH, flattened). The bank width WIDTH stays bound by every similarity_cli element
# provenance and by median_pooling.
AGG_FEATURE_DIM = {'median_pooling': F.WIDTH, 'sampler': 10 * F.WIDTH}
TIE_ORDER_COLS = ('order_sensitive_1', 'order_sensitive_3', 'order_sensitive_5', 'order_sensitive_10')
RR5_SPEC_NOTE = ('RR-5 is driven through its full_spec route (contract M4.3), so its internal COMPLETE.json records '
                 'spec_is_contract: false; this package COMPLETE binds the V2-main contract instead [Q2]')


class GateError(Exception):
    pass


def n_bob_elements(n=None):
    n = F.N_SLIDES if n is None else n
    return -(-n // BOB_BATCH)


def bob_rows(b, n=None):
    n = F.N_SLIDES if n is None else n
    return list(range(BOB_BATCH * (b - 1) + 1, min(BOB_BATCH * b, n) + 1))


def write_new(path, data):
    if os.path.lexists(path):
        raise GateError(f'refusing to overwrite {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'xb') as fh:
        fh.write(data if isinstance(data, bytes) else data.encode())


def jdump(obj):
    return json.dumps(obj, indent=2, sort_keys=True, default=F._json_default) + '\n'


PACKAGE_OUTCOMES = ('FAILED', 'COMPLETE.json', 'PARTIAL.json')
# specification bootstrap mode (M5.2 / A1); the same literals as the adapter (DRAWS_SHA) and compare (BOOTSTRAP_MODE)
CONTRACT_DRAWS_SHA = 'f81586230703cc578752a9b0a9363ddd4219f9cee28287b164b4115eb44a60a2'          # #46
CONTRACT_N_REPS = 5000
CONTRACT_BOOTSTRAP_MODE = 'whole-patient multiplicity weights, n_reps = 5000, draws #46'


def write_failed(run_dir, first_failing_gate, evidence):
    for name in PACKAGE_OUTCOMES:
        if os.path.lexists(os.path.join(run_dir, name)):
            raise GateError(f'{name} already exists in {run_dir}; a new attempt uses a new stage')
    write_new(os.path.join(run_dir, 'FAILED'), jdump({'first_failing_gate': first_failing_gate,
                                                      'evidence_path': evidence}))


def guard_run_dir(run_dir, allow_mismatch=False):
    """ADDENDUM_002 F4 entry guard for m3/m4-ranks/tie-hosts: GateError (nothing written) if the basename of the
    run dir (as given or resolved) is not RUN_STAGE without `allow_mismatch`, or if FAILED already exists there.
    Returns the record stored as `run_dir_guard` in the gate report."""
    resolved = os.path.realpath(run_dir)
    names = {os.path.basename(os.path.normpath(run_dir)), os.path.basename(resolved)}
    if names != {RUN_STAGE} and not allow_mismatch:
        raise GateError(f'--run-dir {run_dir} (resolved {resolved}) is not the {RUN_STAGE} stage; '
                        'pass --allow-run-dir-mismatch only for a deliberate non-production run dir')
    if os.path.lexists(os.path.join(run_dir, 'FAILED')):
        raise GateError(f'{run_dir} already holds FAILED (final); a new attempt uses a new stage')
    return {'run_dir': os.path.abspath(run_dir), 'run_dir_resolved': resolved, 'run_stage': RUN_STAGE,
            'run_dir_mismatch_allowed': names != {RUN_STAGE}}


def _load_rebuild(p):
    """ADDENDUM_002 F8: (record, None) for a well-formed m1_rebuild_<m>.json ({} if absent), else (None, reason).
    A missing `provenance` key is not malformed (its feature_dim is then unrecorded -> the consistency check fails)."""
    if not os.path.exists(p):
        return {}, None
    try:
        with open(p) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        return None, f'unreadable ({type(e).__name__}: {e})'
    if not isinstance(d, dict):
        return None, f'top level is {type(d).__name__}, not an object'
    if 'provenance' in d and d['provenance'] is not None and not isinstance(d['provenance'], dict):
        return None, f'provenance is {type(d["provenance"]).__name__}, not an object'
    try:
        hash(d.get('same_patient_pairs_masked'))
    except TypeError:
        return None, f'same_patient_pairs_masked is {type(d["same_patient_pairs_masked"]).__name__} (unhashable)'
    return d, None


def _load_pickle(p):
    with open(p, 'rb') as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------------------------------
# M3.5.1 rows, element outcomes, census, provenance count
# ----------------------------------------------------------------------------------------------

def element_outcomes(run_dir, kind, n_elements):
    """Latest attempt per element across canary+fanout jobnames; each must have an LSF epilogue with rc 0."""
    latest = {}
    for jn in JOBS[kind]:
        for idx, (jid, p) in F.latest_attempts(run_dir, jn).items():
            if idx not in latest or jid > latest[idx][1]:
                latest[idx] = (jn, jid, p)
    bad = []
    rows = {}
    for idx in range(1, n_elements + 1):
        if idx not in latest:
            bad.append(f'{kind} element {idx}: no LSF out file')
            continue
        jn, jid, p = latest[idx]
        lsf = F.parse_lsf_out(open(p, errors='replace').read())
        rows[idx] = {'job': jn, 'job_id': jid, **lsf}
        if lsf['rc'] != 0:
            bad.append(f'{kind} element {idx} (job {jid}): rc {lsf["rc"]}')
    extra = sorted(set(latest) - set(range(1, n_elements + 1)))
    if extra:
        bad.append(f'{kind}: unexpected element indices {extra[:10]}')
    return rows, latest, bad


def census_rows(run_dir, kind, latest):
    """M1.3 census rule on every element (latest attempt binds; superseded attempts are reported, not gated)."""
    rows = F.stderr_census(run_dir, JOBS[kind])
    keep = {(jn, jid, idx) for idx, (jn, jid, _) in latest.items()}
    for r in rows:
        r['binding'] = (r['job'], r['job_id'], r['index']) in keep
    return rows


def row_files(sim_dir, bank_ids):
    """Each row: list of len(bank) (id, distance) pairs over the bank id set; byte sizes recorded."""
    want = set(bank_ids)
    files = sorted(glob.glob(os.path.join(sim_dir, 'similarity_*.pkl')))
    ids = [os.path.basename(p)[len('similarity_'):-len('.pkl')] for p in files]
    bad = []
    if sorted(ids) != sorted(bank_ids):
        bad.append(f'{sim_dir}: row file ids != bank ids ({len(ids)} files; missing '
                   f'{sorted(want - set(ids))[:5]}, extra {sorted(set(ids) - want)[:5]})')
    sizes = []
    for qid, p in zip(ids, files):
        sizes.append(os.path.getsize(p))
        row = _load_pickle(p)
        if not isinstance(row, list) or len(row) != len(bank_ids):
            bad.append(f'{p}: not a list of {len(bank_ids)} pairs')
            continue
        try:
            got = [str(o) for o, _ in row]
        except (TypeError, ValueError):
            bad.append(f'{p}: entries are not (id, distance) pairs')
            continue
        if len(set(got)) != len(got) or set(got) != want:
            bad.append(f'{p}: row id set != bank ids')
    return {'n_files': len(files), 'bytes_min': min(sizes) if sizes else None,
            'bytes_max': max(sizes) if sizes else None}, bad


def probe_row_bytes(fidelity_dir, kind):
    p = os.path.join(fidelity_dir, f'm1_{kind}_probe.json')
    if not os.path.exists(p):
        return None
    per = json.load(open(p)).get('per_row') or []
    b = [r['row_pickle_bytes'] for r in per if r.get('row_pickle_bytes') is not None]
    return {'bytes_min': min(b), 'bytes_max': max(b), 'n': len(b), 'source': p} if b else None


def provenance_names(prov_dir, n):
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(prov_dir, 'provenance_*.json')))
    want = sorted(f'provenance_{i}.json' for i in range(1, n + 1))
    return have == want, have


# ----------------------------------------------------------------------------------------------
# M3.5.2-M3.5.4 matrices, sentinels, provenance fields
# ----------------------------------------------------------------------------------------------

def matrix_structure(df, ref_index, ref_cols, n):
    """2.  n x n float64 DataFrame, str ids equal to the published order, no NaN/inf."""
    import pandas as pd
    out = {'is_dataframe': isinstance(df, pd.DataFrame)}
    if not out['is_dataframe']:
        return out, None
    a = df.to_numpy()
    out.update({'shape_ok': df.shape == (n, n), 'float64': all(str(t) == 'float64' for t in df.dtypes) and
                a.dtype == np.float64, 'index_equal_published_str': [str(i) for i in df.index] == ref_index,
                'columns_equal_published_str': [str(c) for c in df.columns] == ref_cols,
                'no_nan': not bool(np.isnan(a).any()), 'no_inf': not bool(np.isinf(a).any())})
    return out, a


def sentinel_unmasked(a):
    """3. cells >= 1e6 exactly the diagonal; off-diagonal finite, < 1e6, >= 0."""
    n = a.shape[0]
    ge = a >= F.SENTINEL
    off = ~np.eye(n, dtype=bool)
    return {'sentinel_exactly_diagonal': bool(np.array_equal(ge, np.eye(n, dtype=bool))),
            'off_diagonal_finite': bool(np.isfinite(a[off]).all()),
            'off_diagonal_lt_1e6': bool((a[off] < F.SENTINEL).all()),
            'off_diagonal_ge_0': bool((a[off] >= 0).all()), 'n_ge_1e6': int(ge.sum())}


def sentinel_masked(a, pub):
    ge = a >= F.SENTINEL
    gp = pub >= F.SENTINEL
    return {'pattern_equal_published_masked': bool(np.array_equal(ge, gp)), 'n_ge_1e6': int(ge.sum()),
            'n_ge_1e6_published': int(gp.sum())}


def matrix_path(run_dir, m, masked):
    return os.path.join(run_dir, 'similarity_matrix',
                        f'combined_similarity_matrix_{MODEL}_{m}{"_no_same_patient" if masked else ""}.pkl')


def provenance_paths(run_dir):
    p = {m: os.path.join(run_dir, 'logs', 'aggregate', f'provenance_{MODEL}_{m}.json') for m in AGG_METHODS}
    for m in ('ot_cosine', 'yottixel'):
        p[m] = os.path.join(run_dir, 'logs', 'combine_similarity', f'provenance_{MODEL}_{m}.json')
    return p


def provenance_checks(m, prov, n, rebuild_pairs, rebuild_feature_dim=None):
    """4. (+ the M3.5.3 same_patient_pairs_masked rule). aggregate_cli records n_slides/feature_dim;
    combine_similarity_cli records n_query/n_reference/square (see IMPLEMENTATION_REPORT ambiguity).
    AMENDMENT_004: aggregate feature_dim must equal AGG_FEATURE_DIM[m] (feature_dim_matches_method) and the
    accepted fidelity rebuild's provenance.feature_dim (fidelity_rebuild_feature_dim_consistent)."""
    c = {'same_patient_pairs_masked_eq_rebuild': rebuild_pairs is not None and
         prov.get('same_patient_pairs_masked') == rebuild_pairs}
    if m in AGG_METHODS:
        fd = prov.get('feature_dim')
        c.update({'n_slides_eq_n': prov.get('n_slides') == n,
                  'feature_dim_matches_method': fd is not None and fd == AGG_FEATURE_DIM[m],
                  'fidelity_rebuild_feature_dim_consistent': rebuild_feature_dim is not None and fd is not None and
                  fd == rebuild_feature_dim,
                  'model_is_fp32': prov.get('model') == MODEL})
    else:
        c.update({'n_query_eq_n': prov.get('n_query_slides') == n, 'n_reference_eq_n':
                  prov.get('n_reference_slides') == n, 'square_true': prov.get('square') is True,
                  'model_is_fp32': prov.get('model') == MODEL})
    return c


def similarity_provenance_checks(prov_dir, n_elements, fp32_dir, n, kind):
    bad = []
    for i in range(1, n_elements + 1):
        p = os.path.join(prov_dir, f'provenance_{i}.json')
        if not os.path.exists(p):
            continue                       # counted by provenance_names
        d = json.load(open(p))
        rng = [i, i] if kind == 'ot' else [bob_rows(i, n)[0], bob_rows(i, n)[-1]]
        want = {'feature_dim': F.WIDTH, 'query_features': fp32_dir, 'n_library': n, 'query_range': rng,
                'encoder': MODEL, 'method': kind, 'external': False, 'num_workers': 1, 'lsf_index': str(i)}
        for k, v in want.items():
            if d.get(k) != v:
                bad.append(f'{p}: {k}={d.get(k)!r} != {v!r}')
    return bad


# ----------------------------------------------------------------------------------------------
# A9 host census, run resources
# ----------------------------------------------------------------------------------------------

HOST_COLS = ('job', 'job_id', 'index', 'hostname', 'cpu_model_name', 'os_release', 'lsb_queue', 'file')


def host_rows(stage_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(stage_dir, 'host_sidecar', '*.json'))):
        base = os.path.basename(p)[:-len('.json')]
        job, _, rest = base.rpartition('.')
        jid, _, idx = rest.partition('_')
        d = json.load(open(p))
        rows.append({'job': job, 'job_id': int(jid) if jid.isdigit() else jid, 'index': int(idx) if idx.isdigit()
                     else idx, 'hostname': d.get('hostname'), 'cpu_model_name': d.get('cpu_model_name'),
                     'os_release': d.get('os_release'), 'lsb_queue': d.get('lsb_queue'), 'file': p})
    return rows


def tsv(rows, cols):
    return '\t'.join(cols) + '\n' + ''.join('\t'.join(str(r.get(c, '')) for c in cols) + '\n' for r in rows)


def resources(stage_dir):
    """M0.6 run_resources.json: every LSF out file of this stage (all attempts)."""
    out = []
    for p in sorted(glob.glob(os.path.join(stage_dir, 'lsf', 'v2m_*.out'))):
        base = os.path.basename(p)[:-len('.out')]
        job, _, rest = base.rpartition('.')
        jid, _, idx = rest.partition('_')
        d = F.parse_lsf_out(open(p, errors='replace').read())
        out.append({'job': job, 'job_id': jid, 'index': idx, 'file': p, **{k: d[k] for k in (
            'rc', 'run_time_s', 'max_memory_mb', 'host', 'queue', 'term_reason')}})
    summ = {}
    for r in out:
        s = summ.setdefault(r['job'], {'n': 0, 'max_memory_mb': None, 'max_run_time_s': None, 'queues': set(),
                                       'hosts': set()})
        s['n'] += 1
        for k, src in (('max_memory_mb', 'max_memory_mb'), ('max_run_time_s', 'run_time_s')):
            if r[src] is not None:
                s[k] = r[src] if s[k] is None else max(s[k], r[src])
        if r['queue']:
            s['queues'].add(r['queue'])
        if r['host']:
            s['hosts'].add(r['host'])
    for s in summ.values():
        s['queues'], s['hosts'] = sorted(s['queues']), len(s['hosts'])
    return {'elements': out, 'per_job': summ}


# ----------------------------------------------------------------------------------------------
# m3 driver
# ----------------------------------------------------------------------------------------------

def run_lineage_bindings(run_dir, amendment=None, amendment_sha=None):
    """AMENDMENT_004: {amendment_004, run_001_failed, lineage_file} ({path, sha256} each) when `run_dir` holds
    LINEAGE_RUN_002.json; None otherwise. A run dir named run_002 without the lineage file, an amendment whose sha256
    differs from the bound value or from the lineage record, or a run_001/FAILED whose sha256 differs from the
    lineage record raise GateError (STOP before anything is written)."""
    lp = os.path.join(run_dir, F.LINEAGE_RUN_002)
    if not os.path.isfile(lp):
        if os.path.basename(os.path.normpath(run_dir)) == os.path.basename(F.RUN_DIR_002):
            raise GateError(f'{lp} missing: run_002 must be created by init-run-002 (AMENDMENT_004)')
        return None
    amendment = amendment or F.AMENDMENT_004
    amendment_sha = amendment_sha or F.AMENDMENT_004_SHA
    try:
        lin = json.load(open(lp))
        rec_amend, rec_failed = lin['amendment'], lin['run_001_failed']
    except (ValueError, KeyError, TypeError) as e:
        raise GateError(f'{lp} unreadable: {e}')
    if not os.path.isfile(amendment):
        raise GateError(f'AMENDMENT_004 {amendment} missing')
    a_sha = F.sha256_file(amendment)
    if a_sha != amendment_sha or a_sha != rec_amend.get('sha256'):
        raise GateError(f'AMENDMENT_004 sha256 {a_sha} != bound {amendment_sha} / lineage {rec_amend.get("sha256")}')
    fp = rec_failed.get('path')
    if not fp or not os.path.isfile(fp):
        raise GateError(f'run_001 FAILED {fp} missing')
    f_sha = F.sha256_file(fp)
    if f_sha != rec_failed.get('sha256'):
        raise GateError(f'run_001 FAILED sha256 {f_sha} != lineage {rec_failed.get("sha256")}')
    return {'amendment_004': {'path': os.path.abspath(amendment), 'sha256': a_sha},
            'run_001_failed': {'path': fp, 'sha256': f_sha},
            'lineage_file': {'path': os.path.abspath(lp), 'sha256': F.sha256_file(lp)}}


def m3_gates(run_dir, fidelity_dir, fp32_dir, pub_paths=None, verify_published=True, amendment=None):
    n = F.N_SLIDES
    rep = {'schema': 'r2_1_virchow2_fp32_main.m3_matrix_gates.v1', 'contract_sha256': CONTRACT_SHA, 'gates': {}}
    lb = run_lineage_bindings(run_dir, amendment)
    if lb is not None:
        rep.update(lb)
    fails = []
    bank_ids = F.sorted_pkl_ids(fp32_dir)
    if len(bank_ids) != n:
        fails.append(f'bank {fp32_dir}: {len(bank_ids)} pkl != {n}')

    fe = os.path.join(run_dir, 'failed_elements.jsonl')
    if os.path.exists(fe) and open(fe).read().strip():
        fails.append('M0.5: failed_elements.jsonl is non-empty (an element FAILED; no partial matrix)')

    # ---- 1. rows, outcomes, census, provenance count ----
    g1 = {}
    census_all, sidecar_missing = [], []
    hosts = {(r['job'], r['job_id'], r['index']): r for r in host_rows(run_dir)}
    for kind, n_el in (('ot', n), ('bob', n_bob_elements(n))):
        tag = F.SIM_TAG[kind]
        outcomes, latest, bad = element_outcomes(run_dir, kind, n_el)
        fails += bad
        cen = census_rows(run_dir, kind, latest)
        census_all += cen
        binding = [r for r in cen if r['binding']]
        if len(binding) != n_el:
            fails.append(f'{kind}: stderr files for {len(binding)} of {n_el} elements')
        tb = [(r['job_id'], r['index']) for r in binding if r['Traceback']]
        if tb:
            fails.append(f'{kind}: Traceback in element stderr {tb[:5]}')
        nit = [(r['job_id'], r['index'], r['numItermax_lines']) for r in binding if r['numItermax']]
        for idx, (jn, jid, _) in latest.items():
            if (jn, jid, idx) not in hosts:
                sidecar_missing.append(f'{jn}.{jid}_{idx}')
        sim_dir = os.path.join(run_dir, 'similarity', f'{MODEL}_{kind}')
        rf, bad = row_files(sim_dir, bank_ids)
        fails += bad
        probe = probe_row_bytes(fidelity_dir, kind)
        agree = None if probe is None or rf['bytes_min'] is None else (
            rf['bytes_min'] >= probe['bytes_min'] and rf['bytes_max'] <= probe['bytes_max'])
        failed_lines = [] if kind == 'ot' else sum((F.failed_stdout_lines(run_dir, jn) for jn in JOBS['bob']), [])
        if failed_lines:
            fails.append(f'bob: FAILED line(s) in stdout {failed_lines[:3]}')
        prov_dir = os.path.join(run_dir, 'logs', f'similarity_{MODEL}_{kind}')
        ok_names, have = provenance_names(prov_dir, n_el)
        if not ok_names:
            fails.append(f'{kind}: provenance files != provenance_1..{n_el}.json ({len(have)} present)')
        pbad = similarity_provenance_checks(prov_dir, n_el, fp32_dir, n, kind)
        fails += pbad[:50]
        g1[tag] = {'n_elements': n_el, 'element_failures': bad, 'row_files': rf, 'probe_row_bytes': probe,
                   'row_bytes_within_probe_range': agree,
                   'row_bytes_note': 'reported, not gated: "must agree ... or be explained" (M3.5.1)',
                   'failed_stdout_lines': failed_lines, 'traceback_elements': tb, 'numItermax_elements': nit,
                   'n_provenance': len(have), 'provenance_names_ok': ok_names, 'provenance_field_failures': pbad[:50],
                   'max_memory_mb': max((o['max_memory_mb'] or 0) for o in outcomes.values()) if outcomes else None}
    if sidecar_missing:
        fails.append(f'A9: no host sidecar for {len(sidecar_missing)} similarity elements {sidecar_missing[:5]}')
    rep['gates']['m3_5_1_rows'] = g1

    # ---- 2-4. matrices ----
    rebuild, rebuild_fd, rebuild_malformed = {}, {}, {}
    for m in AGG_METHODS:
        p = os.path.join(fidelity_dir, f'm1_rebuild_{m}.json')
        d, why = _load_rebuild(p)
        if d is None:                                                         # ADDENDUM_002 F8
            rebuild_malformed[m] = {'path': p, 'reason': why}
            fails.append(f'M1.2 rebuild {m}: {p} malformed: {why}')
            d = {}
        rebuild[m] = d.get('same_patient_pairs_masked')
        rebuild_fd[m] = (d.get('provenance') or {}).get('feature_dim')       # AMENDMENT_004
    rb_vals = {v for v in rebuild.values()}
    rebuild_pairs = rebuild['median_pooling'] if len(rb_vals) == 1 else None
    if rebuild_pairs is None:
        fails.append(f'M1.2 rebuild same_patient_pairs_masked unavailable/inconsistent: {rebuild}')
    pub_paths = pub_paths or {k: os.path.join(F.PUBM, f'combined_similarity_matrix_virchow2_{k}.pkl')
                              for k in F.PUB_MATRIX}
    provs = provenance_paths(run_dir)
    g2, g4, g5 = {}, {}, {}
    for m in F.METHODS:
        for masked in (False, True):
            key = f'{m}_no_same_patient' if masked else m
            pp = pub_paths[key]
            if verify_published:
                F.require_binding(pp, F.PUB_MATRIX[key][2], f'#{F.PUB_MATRIX[key][0]}')
            pub = _load_pickle(pp)
            ref_index, ref_cols = [str(i) for i in pub.index], [str(c) for c in pub.columns]
            mp = matrix_path(run_dir, m, masked)
            if not os.path.exists(mp):
                fails.append(f'{mp} missing')
                continue
            df = _load_pickle(mp)
            st, a = matrix_structure(df, ref_index, ref_cols, n)
            if a is not None:
                st.update(sentinel_masked(a, pub.to_numpy()) if masked else sentinel_unmasked(a))
                g5[key] = {k: v for k, v in F.compare_frames(df, pub).items()
                           if k in ('n_live', 'max_abs', 'max_rel', 'n_differing_cells')}
                live = (a < F.SENTINEL) & (pub.to_numpy() < F.SENTINEL)
                d = np.abs(a[live] - pub.to_numpy()[live])
                g5[key]['median_abs'] = float(np.median(d)) if d.size else None
            st['sha256'] = F.sha256_file(mp)
            g2[key] = st
            fails += [f'M3.5.2/3 {key}: {k}' for k, v in st.items() if v is False]
        pv = json.load(open(provs[m])) if os.path.exists(provs[m]) else {}
        if not pv:
            fails.append(f'M3.5.4 {m}: provenance {provs[m]} missing')
        c = provenance_checks(m, pv, n, rebuild_pairs, rebuild_fd.get(m))
        g4[m] = {'path': provs[m], 'checks': c, 'same_patient_pairs_masked': pv.get('same_patient_pairs_masked')}
        if m in AGG_METHODS:
            g4[m].update({'feature_dim_expected': AGG_FEATURE_DIM[m], 'feature_dim_observed': pv.get('feature_dim'),
                          'feature_dim_matches_method': c['feature_dim_matches_method'],
                          'fidelity_rebuild_feature_dim': rebuild_fd.get(m),
                          'fidelity_rebuild_path': os.path.join(fidelity_dir, f'm1_rebuild_{m}.json')})
        fails += [f'M3.5.3/4 {m}: {k}' for k, v in c.items() if not v]
    rep['gates'].update({'m3_5_2_3_matrices': g2, 'm3_5_4_provenance': g4, 'rebuild_same_patient_pairs': rebuild,
                         'rebuild_feature_dim': rebuild_fd, 'm3_5_5_descriptive_vs_published': g5})
    if rebuild_malformed:
        rep['gates']['rebuild_malformed'] = rebuild_malformed
    rep.update({'census_keys': list(F.CENSUS_KEYS), 'failures': fails, 'pass': not fails,
                'first_failing_gate': fails[0] if fails else None})
    return rep, census_all


def cmd_m3(a):
    run_dir = os.path.abspath(a.run_dir)
    guard = guard_run_dir(run_dir, getattr(a, 'allow_run_dir_mismatch', False))      # ADDENDUM_002 F4
    amendment = getattr(a, 'amendment', None)
    if amendment is None:
        rep, census = m3_gates(run_dir, os.path.abspath(a.fidelity_dir), a.fp32)
    else:
        rep, census = m3_gates(run_dir, os.path.abspath(a.fidelity_dir), a.fp32, amendment=amendment)
    rep['run_dir_guard'] = guard
    write_new(os.path.join(run_dir, 'stderr_census.tsv'), tsv(
        [{**r, 'binding': r['binding']} for r in census], ('job', 'job_id', 'index') + F.CENSUS_KEYS +
        ('binding', 'file')))
    write_new(os.path.join(run_dir, 'host_census.tsv'), tsv(host_rows(run_dir), HOST_COLS))
    write_new(os.path.join(run_dir, 'run_resources.json'), jdump(resources(run_dir)))
    write_new(os.path.join(run_dir, 'm3_matrix_gates.json'), jdump(rep))
    if not rep['pass']:
        write_failed(run_dir, rep['first_failing_gate'], os.path.join(run_dir, 'm3_matrix_gates.json'))
        print(f'STOP: {rep["first_failing_gate"]}', file=sys.stderr)
        return 1
    print(json.dumps({'pass': True}))
    return 0


# ----------------------------------------------------------------------------------------------
# M4.1 ranks resolution
# ----------------------------------------------------------------------------------------------

def _metrics_matrix_field(d):
    """ADDENDUM_003 §2.3: (path, key) of the metrics_cli input matrix. The accepted metrics_cli writes it under
    `config.input_similarity_matrix`; a top-level `input_similarity_matrix` is the fallback; (None, None) if neither
    is a non-empty string."""
    cfg = d.get('config') if isinstance(d, dict) else None
    v = cfg.get('input_similarity_matrix') if isinstance(cfg, dict) else None
    if isinstance(v, str) and v:
        return v, 'config'
    v = d.get('input_similarity_matrix') if isinstance(d, dict) else None
    if isinstance(v, str) and v:
        return v, 'top_level'
    return None, None


def stage_lineage(run_dir):
    """ADDENDUM_003: the run-lineage record of `run_dir` itself (F.RUN_LINEAGES file whose schema matches and whose
    dest_dir is run_dir), as (path, record); (None, None) if none. A lineage file carried from the source stage
    (run_003/LINEAGE_RUN_002.json, dest_dir run_002) is never selected."""
    hits = []
    for name, schema in F.RUN_LINEAGES.values():
        p = os.path.join(run_dir, name)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        try:
            lin = json.load(open(p))
            same = lin.get('schema') == schema and isinstance(lin.get('dest_dir'), str) and \
                os.path.samefile(lin['dest_dir'], run_dir)
        except (ValueError, OSError, AttributeError):
            continue
        if same:
            hits.append((p, lin))
    return hits[0] if len(hits) == 1 else (None, None)


def lineage_equivalent(run_dir, want, got, lineage):
    """ADDENDUM_003 (implementation_008, independent ratification required): a carried metrics_cli output names the
    SOURCE stage's copy of a file (run_002/...) while M4.1 expects the run_003 copy. Accept `got` for `want` only if
    `lineage` = (path, record) is run_dir's own run-lineage record and: rel = relpath(want, run_dir) stays inside
    run_dir; realpath(got) == realpath(source_dir/rel); rel is a listed, non-append-only lineage file; and
    sha256(want) == its sha256_dest, sha256(got) == its sha256_source (equal by construction). One hop only.
    Returns (ok, record)."""
    lp, lin = lineage
    rec = {'route': 'lineage', 'lineage_file': None, 'ok': False}
    if lp is None or got is None:
        rec['why'] = 'no own run-lineage record' if lp is None else 'no path recorded'
        return False, rec
    rec['lineage_file'] = F.binding(lp)
    rec['lineage_schema'] = lin.get('schema')
    rel = os.path.relpath(os.path.abspath(want), os.path.abspath(run_dir)).replace(os.sep, '/')
    rec['rel'] = rel
    if rel == '..' or rel.startswith('../') or os.path.isabs(rel):
        rec['why'] = 'expected path outside the run dir'
        return False, rec
    src = lin.get('source_dir')
    if not isinstance(src, str):
        rec['why'] = 'lineage source_dir missing'
        return False, rec
    src_path = os.path.join(src, rel)
    rec['source_path'] = src_path
    if os.path.realpath(got) != os.path.realpath(src_path):
        rec['why'] = 'recorded path is not the lineage source copy of the expected path'
        return False, rec
    ent = [e for e in lin.get('files', []) if isinstance(e, dict) and e.get('path') == rel]
    if len(ent) != 1 or ent[0].get('append_only'):
        rec['why'] = 'expected path not a (single, non-append-only) lineage-listed file'
        return False, rec
    e = ent[0]
    rec.update({'mode': e.get('mode'), 'sha256_source_recorded': e.get('sha256_source'),
                'sha256_dest_recorded': e.get('sha256_dest')})
    if not (os.path.isfile(want) and os.path.isfile(got)):
        rec['why'] = 'expected or recorded file missing'
        return False, rec
    sw, sg = F.sha256_file(want), F.sha256_file(got)
    rec.update({'sha256_expected': sw, 'sha256_recorded_path': sg})
    try:
        rec['samefile_informational'] = os.path.samefile(want, got)
    except OSError:
        rec['samefile_informational'] = False
    if not (sw == e.get('sha256_dest') and sg == e.get('sha256_source') and sw == sg):
        rec['why'] = 'sha256 differs from the lineage record'
        return False, rec
    rec['ok'] = True
    return True, rec


def _path_resolution(run_dir, want, got, lineage):
    """(ok, record) for a recorded path against the expected run-stage path: route `direct` (realpath equal, the
    pre-ADDENDUM_003 rule) else route `lineage` (lineage_equivalent)."""
    if got is not None and os.path.realpath(got) == os.path.realpath(want):
        return True, {'route': 'direct', 'ok': True}
    return lineage_equivalent(run_dir, want, got, lineage)


def _script_hashes():
    """ADDENDUM_003 §2.3 / review_007 Q3: the scripts that produced this gate report."""
    return {'gates': F.binding(os.path.abspath(__file__)), 'fidelity': F.binding(os.path.abspath(F.__file__))}


def _step_receipt(run_dir, step):
    """Bindings of <run>/jobs/<step>/{launch.sha256,step.json} (the script hashes each step ran under, review_007 Q3),
    with launch.sha256 parsed into {path: sha256}; None entries if absent."""
    jd = os.path.join(run_dir, 'jobs', step)
    out = {'step': step}
    for n in ('launch.sha256', 'step.json'):
        p = os.path.join(jd, n)
        out[n] = F.binding(p) if os.path.isfile(p) else None
    lp = os.path.join(jd, 'launch.sha256')
    if os.path.isfile(lp):
        lines = {}
        for ln in open(lp, errors='replace').read().splitlines():
            parts = ln.split(None, 1)
            if len(parts) == 2 and re.fullmatch(r'[0-9a-f]{64}', parts[0]):
                lines[parts[1].strip()] = parts[0]
        out['launch_hashes'] = lines
    sp = os.path.join(jd, 'step.json')
    if os.path.isfile(sp):
        try:
            sj = json.load(open(sp))
            out['step_stage'] = sj.get('stage')
            out['step_args'] = sj.get('args')
            out['step_module'] = sj.get('module')
        except (ValueError, AttributeError):
            out['step_json_unreadable'] = True
    return out


def m4_ranks_resolution(run_dir):
    """M4.1 per method. ADDENDUM_003: the matrix path is read from `config.input_similarity_matrix` (accepted
    metrics_cli shape), else the top-level key (`input_similarity_matrix_key` records which; null = neither -> the
    check fails); the full `config` block is recorded. A recorded path (matrix or `ranked_lists`) is accepted when it
    resolves to the expected run-stage file (route `direct`) or, for outputs carried by lineage from the source stage,
    when it is that stage's lineage-recorded, hash-identical copy (route `lineage`, see lineage_equivalent). The report
    records the gates/fidelity script hashes and each m4_ranks step's launch.sha256/step.json binding."""
    lineage = stage_lineage(run_dir)
    rep, fails = {'methods': {}, 'script_sha256': _script_hashes(),
                  'gate_step': _step_receipt(run_dir, 'm4_ranks_gate'),
                  'stage_lineage': F.binding(lineage[0]) if lineage[0] else None}, []
    for m in F.METHODS:
        mj = os.path.join(run_dir, 'c2_metric_suite', f'metrics_{MODEL}_{m}.json')
        want = matrix_path(run_dir, m, True)
        d = json.load(open(mj)) if os.path.exists(mj) else {}
        got, key = _metrics_matrix_field(d)
        cfg = d.get('config') if isinstance(d, dict) else None
        top = d.get('input_similarity_matrix') if isinstance(d, dict) else None
        ranks = os.path.join(run_dir, 'c2_metric_suite', f'ranked_lists_{MODEL}_{m}.csv.gz')
        stdout_hits = []
        for jid, idx, p in F.lsf_logs(run_dir, f'v2m_m4_ranks_{m}', 'out'):
            txt = F.program_stdout(open(p, errors='replace').read())
            stdout_hits += [ln.strip() for ln in txt.splitlines() if 'similarity_matrix' in ln][:5]
        mat_ok, mat_route = _path_resolution(run_dir, want, got, lineage)
        rl = d.get('ranked_lists') if isinstance(d, dict) else None
        rl = rl if isinstance(rl, str) and rl else None
        rl_ok, rl_route = _path_resolution(run_dir, ranks, rl, lineage) if rl is not None else \
            (False, {'route': None, 'ok': False, 'why': 'no ranked_lists field'})
        c = {'metrics_json_present': bool(d), 'resolved_is_fp32_run_matrix': got is not None and mat_ok,
             'not_published_tree': got is not None and not os.path.realpath(got).startswith(
                 os.path.realpath(F.PUBM)),
             'ranks_csv_gz_present': os.path.exists(ranks),
             'no_parquet': not os.path.exists(ranks[:-len('.csv.gz')] + '.parquet'),
             'ranked_lists_field_matches': rl is not None and rl_ok}
        if c['resolved_is_fp32_run_matrix']:
            c['matrix_sha256'] = F.sha256_file(want)
        rec = {'metrics_json': mj, 'metrics_json_sha256': F.sha256_file(mj) if os.path.exists(mj) else None,
               'input_similarity_matrix': got, 'input_similarity_matrix_key': key,
               'config': cfg, 'expected': want, 'matrix_resolution': mat_route,
               'ranked_lists': ranks, 'ranked_lists_field': rl, 'ranked_lists_resolution': rl_route,
               'ranked_lists_sha256': F.sha256_file(ranks) if os.path.exists(ranks) else None,
               'producing_step': _step_receipt(run_dir, f'm4_ranks_{m}'),
               'stdout_lines': stdout_hits, 'checks': c,
               'note': 'metrics_*.json are produced, not used; not a reported result [m11]'}
        if key == 'config' and isinstance(top, str) and top and top != got:
            rec['top_level_input_similarity_matrix_ignored'] = top
        rep['methods'][m] = rec
        fails += [f'M4.1 {m}: {k}' for k, v in c.items() if v is False]
    rep.update({'failures': fails, 'pass': not fails})
    return rep


def cmd_m4_ranks(a):
    run_dir = os.path.abspath(a.run_dir)
    guard = guard_run_dir(run_dir, getattr(a, 'allow_run_dir_mismatch', False))      # ADDENDUM_002 F4
    rep = m4_ranks_resolution(run_dir)
    rep['run_dir_guard'] = guard
    write_new(os.path.join(run_dir, 'm4_ranks_resolution.json'), jdump(rep))
    if not rep['pass']:
        write_failed(run_dir, rep['failures'][0], os.path.join(run_dir, 'm4_ranks_resolution.json'))
        print(f'STOP: {rep["failures"][0]}', file=sys.stderr)
        return 1
    return 0


# ----------------------------------------------------------------------------------------------
# A9 tie-census host diversity
# ----------------------------------------------------------------------------------------------

CELL_JOBS = {'median_pooling': ('v2m_m3_aggregate_median_pooling',), 'sampler': ('v2m_m3_aggregate_sampler',),
             'ot_cosine': JOBS['ot'] + ('v2m_m3_combine_ot',), 'yottixel': JOBS['bob'] + ('v2m_m3_combine_bob',)}


def tie_host_diversity(run_dir, tie_census_csv):
    hosts = host_rows(run_dir)
    out = []
    with open(tie_census_csv, newline='') as fh:
        for r in csv.DictReader(fh):
            if not r['cell_id'].startswith(MODEL + '/'):
                continue
            m = r['cell_id'].split('/', 1)[1]
            sens = {c: int(r[c]) for c in TIE_ORDER_COLS}
            hs = [h for h in hosts if h['job'] in CELL_JOBS[m]]
            out.append({'cell_id': r['cell_id'], 'order_sensitive': any(sens.values()), **sens,
                        'n_elements_with_sidecar': len(hs), 'n_hostnames': len({h['hostname'] for h in hs}),
                        'n_cpu_models': len({h['cpu_model_name'] for h in hs}),
                        'cpu_models': sorted({h['cpu_model_name'] for h in hs if h['cpu_model_name']})})
    return {'source': tie_census_csv, 'source_sha256': F.sha256_file(tie_census_csv), 'cells': out,
            'order_sensitive_cells': [c['cell_id'] for c in out if c['order_sensitive']],
            'note': 'descriptive (A9); host diversity among order-sensitive cells, from the per-element host sidecars'}


def cmd_tie_hosts(a):
    run_dir = os.path.abspath(a.run_dir)
    guard = guard_run_dir(run_dir, getattr(a, 'allow_run_dir_mismatch', False))      # ADDENDUM_002 F4
    src = a.tie_census or os.path.join(run_dir, 'rr5_additive_001', 'tie_census.csv')
    rep = tie_host_diversity(run_dir, src)
    rep['run_dir_guard'] = guard
    write_new(os.path.join(run_dir, 'tie_census_host_diversity.json'), jdump(rep))
    return 0


# ----------------------------------------------------------------------------------------------
# M0.3 stage input re-hash -> <stage>/input_bindings.json
# ----------------------------------------------------------------------------------------------

PS = '/PROSPER_WORKSPACE/prosper-search'
ATTEMPT = os.path.join(F.OUT, 'r2_1_main_grid_comparators_20261011', 'main_virchow2_attempt_001')
I_ROW = re.compile(r'^\| (\d+) \| (.*?) \| (?:[\d,]+ \| )?`([0-9a-f]{64})` \| (.*?) \|\s*$', re.M)
# Entries each stage consumes (implementation reading of M0.3; see IMPLEMENTATION_REPORT). #27/#28 (banks) are
# bound per file by the M1.0/M2 bank check; #70/#71/#74/#75 carry abbreviated hashes in the specification and are
# bound through RR-5's own PIN/pins-file gates and #72.
STAGE_INPUTS = {'fidelity_001': list(range(1, 24)) + list(range(29, 43)) + list(range(62, 66)),
                'run_001': list(range(1, 27)) + list(range(29, 70)) + [72, 73]}
STAGE_INPUTS['run_002'] = list(STAGE_INPUTS['run_001'])     # AMENDMENT_004: same inputs; run_001 is frozen FAILED
STAGE_INPUTS['run_003'] = list(STAGE_INPUTS['run_001'])     # ADDENDUM_003: same inputs; run_002 is frozen FAILED


def contract_inputs(contract=CONTRACT, contract_sha=CONTRACT_SHA):
    """{#: (path, sha256, src)} for every §I row with a full 64-hex sha256."""
    if F.sha256_file(contract) != contract_sha:
        raise GateError('contract sha256 differs from the bound candidate 002')
    txt = open(contract).read()
    rows = {int(m.group(1)): (m.group(2), m.group(3), m.group(4).strip()) for m in I_ROW.finditer(txt)}
    spec43 = None
    out = {}
    for n, (desc, sha, src) in sorted(rows.items()):
        ticks = re.findall(r'`([^`]+)`', desc)
        first = ticks[0] if ticks else ''
        if first.startswith('PS/'):
            path = os.path.join(PS, first[3:])
        elif first.startswith('REPO/'):
            path = os.path.join(F.REPO, first[5:])
        elif first.startswith('OUT/'):
            path = os.path.join(F.OUT, first[4:])
        elif first.startswith('\u2026/main_virchow2_attempt_001/'):
            path = os.path.join(ATTEMPT, first.split('/', 2)[2])
        elif 35 <= n <= 42:
            path = os.path.join(F.PUBM, first)
        elif 'spec source' in desc:
            if spec43 is None:
                spec43 = json.load(open(out[43][0]))
            name = re.search(r'spec source `([^`]+)`', desc).group(1)
            path = spec43['sources'][name]['path']
            if spec43['sources'][name].get('sha256') != sha:
                raise GateError(f'#{n}: #43 source {name} sha256 != contract value')
        else:
            raise GateError(f'#{n}: cannot resolve contract path from {desc[:80]!r}')
        out[n] = (path, sha, src)
    return out


def input_bindings(stage, contract=CONTRACT, contract_sha=CONTRACT_SHA, numbers=None, resolved=None):
    table = resolved if resolved is not None else contract_inputs(contract, contract_sha)
    want = numbers if numbers is not None else STAGE_INPUTS[stage]
    rows, fails = [], []
    for n in want:
        if n not in table:
            fails.append(f'#{n}: no full sha256 row in the contract')
            continue
        path, exp, src = table[n]
        if not os.path.exists(path):
            rows.append({'n': n, 'path': path, 'size': None, 'sha256': None, 'expected_sha256': exp, 'src': src,
                         'match': False})
            fails.append(f'M0.3 #{n}: missing {path}')
            continue
        h = F.sha256_file(path)
        rows.append({'n': n, 'path': path, 'size': os.path.getsize(path), 'sha256': h, 'expected_sha256': exp,
                     'src': src, 'match': h == exp})
        if h != exp:
            fails.append(f'M0.3 #{n}: sha256 mismatch {path}')
    return {'schema': 'r2_1_virchow2_fp32_main.input_bindings.v1', 'stage': stage,
            'contract_sha256': contract_sha, 'inputs': rows,
            'not_rehashed_here': {'27_28': 'banks: per-file sha256 verified by the M1.0/M2 bank check',
                                  '70_71_74_75': 'abbreviated in the contract; bound by RR-5 PIN/pins file and #72'},
            'failures': fails, 'pass': not fails}


def cmd_input_bindings(a):
    stage_dir = os.path.abspath(a.stage_dir or os.path.join(ROOT, a.stage))
    rep = input_bindings(a.stage)
    write_new(os.path.join(stage_dir, 'input_bindings.json'), jdump(rep))
    if not rep['pass']:
        write_failed(stage_dir, rep['failures'][0], os.path.join(stage_dir, 'input_bindings.json'))
        print(f'STOP: {rep["failures"][0]}', file=sys.stderr)
        return 1
    return 0


# ----------------------------------------------------------------------------------------------
# M6 package COMPLETE rule [Q2]
# ----------------------------------------------------------------------------------------------

def _sha_lines(paths, base):
    return ''.join(f'{F.sha256_file(p)}  {os.path.relpath(p, base)}\n' for p in sorted(paths))


def compare_consistency(report_path, rb_dir):
    """F1/F15: the compare report must describe the Run B state that exists now, A3 must have passed (Run B) or be
    pending (no Run B), and the specification bootstrap mode must be asserted. Raises ValueError on any disagreement."""
    r = json.load(open(report_path))
    bad = []
    if r.get('pass') is not True:
        bad.append('compare pass != true')
    a3 = r.get('a3') or {}
    if os.path.exists(rb_dir):
        if r.get('run_b') != 'present':
            bad.append(f"rr5_replacement_001 exists but compare run_b = {r.get('run_b')!r} (compare must be re-run "
                       f"in a new attempt)")
        if a3.get('pass') is not True:
            bad.append('Run B present but A3 pass != true')
        fr = os.path.join(rb_dir, 'faithfulness_report.json')
        bound = (r.get('run_b_faithfulness_report') or {}).get('sha256')
        if not os.path.exists(fr) or bound != F.sha256_file(fr):
            bad.append('compare report does not bind the current Run B faithfulness_report.json')
    else:
        if r.get('run_b') != 'pending':
            bad.append(f"Run B absent but compare run_b = {r.get('run_b')!r}")
        if a3.get('status') != 'pending':
            bad.append('Run B absent but A3 status != pending')
    if r.get('contract_draws') is not True:
        bad.append('contract_draws != true')
    if r.get('n_reps') != CONTRACT_N_REPS:
        bad.append(f"n_reps = {r.get('n_reps')!r} != {CONTRACT_N_REPS}")
    if r.get('draws_sha256') != CONTRACT_DRAWS_SHA:
        bad.append('draws_sha256 != #46')
    if r.get('bootstrap_mode') != CONTRACT_BOOTSTRAP_MODE:
        bad.append(f"bootstrap_mode = {r.get('bootstrap_mode')!r}")
    if bad:
        raise ValueError('; '.join(bad))
    return True


def package_inputs(run_dir, fidelity_dir):
    """Every stage's own outcome; returns (record, failures)."""
    rec, fails = {}, []
    rb = os.path.join(run_dir, 'rr5_replacement_001')

    def need(label, path, check=None):
        ok = os.path.exists(path)
        entry = {'path': path, 'present': ok}
        if ok:
            entry['sha256'] = F.sha256_file(path)
            if check is not None:
                try:
                    ok = bool(check(path))
                except (ValueError, KeyError, TypeError, OSError) as e:
                    entry['error'] = str(e)
                    ok = False
        entry['ok'] = ok
        rec[label] = entry
        if not ok:
            fails.append(f'{label}: {path}')
        return entry

    def jpass(p):
        return json.load(open(p)).get('pass') is True

    need('fidelity_COMPLETE', os.path.join(fidelity_dir, 'COMPLETE'))['fidelity_dir'] = fidelity_dir
    need('m3_matrix_gates', os.path.join(run_dir, 'm3_matrix_gates.json'), jpass)
    need('m4_ranks_resolution', os.path.join(run_dir, 'm4_ranks_resolution.json'), jpass)
    need('eval_attempt_001_COMPLETE', os.path.join(run_dir, 'eval_attempt_001', 'COMPLETE.json'))
    need('rr5_run_a_COMPLETE', os.path.join(run_dir, 'rr5_additive_001', 'COMPLETE.json'),
         lambda p: json.load(open(p)).get('status') == 'COMPLETE')
    for f in ('bit_identity_report.json', 'evaluator_crosscheck.json'):
        need(f'rr5_run_a_{f}', os.path.join(run_dir, 'rr5_additive_001', f), jpass)
    need('comparison_001_report', os.path.join(run_dir, 'comparison_001', 'compare_report.json'),
         lambda p: compare_consistency(p, rb))
    need('materiality', os.path.join(run_dir, 'comparison_001', 'materiality.json'))
    for name in ('FAILED',):
        for d in (run_dir, fidelity_dir, os.path.join(run_dir, 'eval_attempt_001'),
                  os.path.join(run_dir, 'rr5_additive_001'), rb):
            if os.path.lexists(os.path.join(d, name)) or os.path.lexists(os.path.join(d, name + '.json')):
                fails.append(f'{d}: {name} present')
    if os.path.exists(rb):
        need('rr5_run_b_faithfulness', os.path.join(rb, 'faithfulness_report.json'), jpass)
        need('rr5_run_b_raw_COMPLETE', os.path.join(rb, 'raw', 'COMPLETE.json'),
             lambda p: json.load(open(p)).get('status') == 'COMPLETE')
    return rec, fails


def package(run_dir, fidelity_dir, contract=CONTRACT, contract_sha=CONTRACT_SHA, run_b_status=None):
    for name in PACKAGE_OUTCOMES:
        if os.path.lexists(os.path.join(run_dir, name)):
            raise GateError(f'{name} already exists in {run_dir}')
    if F.sha256_file(contract) != contract_sha:
        raise GateError('contract sha256 differs from the bound candidate 002')
    rec, fails = package_inputs(run_dir, fidelity_dir)
    rb = os.path.join(run_dir, 'rr5_replacement_001')
    if os.path.exists(rb):
        run_b = 'COMPLETE' if not any(f.startswith(('rr5_run_b', 'comparison_001_report')) for f in fails) \
            else 'FAILED'
        if run_b_status is not None:
            fails.append(f'--run-b-status {run_b_status} given but rr5_replacement_001 exists')
    else:
        if run_b_status not in ('infeasible', 'not_released'):
            fails.append('Run B absent: --run-b-status infeasible|not_released must be set by the lead')
        run_b = run_b_status
    if fails:
        write_failed(run_dir, fails[0], rec.get(fails[0].split(':')[0], {}).get('path'))
        return {'status': 'FAILED', 'failures': fails}
    # per-directory row manifests, then SHA256SUMS (all files except similarity_<sid>.pkl rows)
    for kind in ('ot', 'bob'):
        d = os.path.join(run_dir, 'similarity', f'{MODEL}_{kind}')
        write_new(os.path.join(run_dir, 'similarity', f'{MODEL}_{kind}.sha256'),
                  _sha_lines(glob.glob(os.path.join(d, 'similarity_*.pkl')), run_dir))
    files = []
    for dp, dn, fn in os.walk(run_dir):
        dn.sort()
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, run_dir)
            if rel in ('SHA256SUMS',) + PACKAGE_OUTCOMES:
                continue
            if os.path.basename(dp).startswith(f'{MODEL}_') and os.path.basename(os.path.dirname(dp)) == \
                    'similarity' and f.startswith('similarity_') and f.endswith('.pkl'):
                continue
            files.append(p)
    write_new(os.path.join(run_dir, 'SHA256SUMS'), _sha_lines(files, run_dir))
    mat = json.load(open(os.path.join(run_dir, 'comparison_001', 'materiality.json')))
    ea = os.path.join(run_dir, 'eval_attempt_001')
    run_a_c = json.load(open(os.path.join(run_dir, 'rr5_additive_001', 'COMPLETE.json')))
    comp = {'schema': 'r2_1_virchow2_fp32_main.package_complete.v1',
            'contract': {'path': contract, 'sha256': contract_sha},
            'stage_outcomes': rec,
            'matrices': {os.path.basename(matrix_path(run_dir, m, k)): F.sha256_file(matrix_path(run_dir, m, k))
                         for m in F.METHODS for k in (False, True)},
            'eval_attempt_001': {os.path.relpath(p, run_dir): F.sha256_file(p)
                                 for p in sorted(glob.glob(os.path.join(ea, '*'))) if os.path.isfile(p)},
            'run_a': 'COMPLETE', 'run_b': run_b,
            'rr5_internal_spec_is_contract': {'run_a': run_a_c.get('spec_is_contract'), 'note': RR5_SPEC_NOTE},
            'package_materiality_flag': mat.get('package_flag'),
            'sha256sums_sha256': F.sha256_file(os.path.join(run_dir, 'SHA256SUMS'))}
    cr = json.load(open(os.path.join(run_dir, 'comparison_001', 'compare_report.json')))
    comp['bootstrap_mode'] = {k: cr.get(k) for k in ('bootstrap_mode', 'n_reps', 'draws_sha256', 'contract_draws')}
    comp['a3'] = 'pass' if run_b == 'COMPLETE' else 'pending'
    if run_b == 'COMPLETE':
        fr = json.load(open(os.path.join(rb, 'faithfulness_report.json')))
        comp['run_b_oracle_winner_count_reproduces_submitted'] = fr.get('oracle_winner_count_reproduces_submitted')
        comp['status'] = 'COMPLETE'
        write_new(os.path.join(run_dir, 'COMPLETE.json'), jdump(comp))
        return {**comp}
    # F4 (design decision): no Run B -> never COMPLETE.json; replacement products stay pending
    comp.update({'status': 'COMPLETE_RUN_A_ONLY', 'replacement_products': 'pending',
                 'note': ('Run B infeasible or not released: no Virchow2 row substitution in cross-cell products '
                          '(contract l.545-547); this is not a complete package')})
    write_new(os.path.join(run_dir, 'PARTIAL.json'), jdump(comp))
    return {**comp}


def cmd_package(a):
    rep = package(os.path.abspath(a.run_dir), os.path.abspath(a.fidelity_dir), run_b_status=a.run_b_status)
    print(json.dumps({'status': rep['status'], 'failures': rep.get('failures', [])[:5]}))
    return 0 if rep['status'] in ('COMPLETE', 'COMPLETE_RUN_A_ONLY') else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('input-bindings')
    p.add_argument('--stage', required=True, choices=sorted(STAGE_INPUTS))
    p.add_argument('--stage-dir', default=None)
    p.set_defaults(fn=cmd_input_bindings)
    for name, fn in (('m3', cmd_m3), ('m4-ranks', cmd_m4_ranks), ('tie-hosts', cmd_tie_hosts),
                     ('package', cmd_package)):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', default=RUN_DIR)
        p.add_argument('--fidelity-dir', default=F.FIDELITY_DIR)
        p.set_defaults(fn=fn)
        if name != 'package':
            p.add_argument('--allow-run-dir-mismatch', action='store_true',
                           help=f'ADDENDUM_002 F4: accept a --run-dir whose basename is not {RUN_STAGE} '
                                '(non-production use only); a run dir holding FAILED is always refused')
        if name == 'm3':
            p.add_argument('--fp32', default=F.FP32)
            p.add_argument('--amendment', default=F.AMENDMENT_004,
                           help='AMENDMENT_004.md (sha256 must equal F.AMENDMENT_004_SHA and the run_002 lineage '
                                'record); used only when the run dir holds LINEAGE_RUN_002.json')
        if name == 'tie-hosts':
            p.add_argument('--tie-census', default=None)
        if name == 'package':
            p.add_argument('--run-b-status', choices=('infeasible', 'not_released'), default=None,
                           help='lead-set when rr5_replacement_001 does not exist')
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (GateError, F.FidelityError) as e:
        print(f'STOP: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
