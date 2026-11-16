"""V2-main (R2.1) RR-5 adapter: Run A (additive, 42 cells) and Run B (replacement view, option (ii)).

Specification: prosper_revision_output/r2_1_virchow2_fp32_main_20261013/contract_candidate_002/SPECIFICATION.md
(f0ef449f...) M4.2/M4.3, LEAD_RESOLUTIONS_002.md (U-1..U-10) and LEAD_RESOLUTIONS_002_ADDENDUM.md (A1-A10).

The frozen RR-5 script (scripts/r2_1_main_grid_plurality.py, sha256 611a502d...) is read as bytes, hash-checked,
compiled and executed in a fresh module namespace. Every statistic, product and gate is RR-5's own function called
from that namespace (A1 / N.15). The only namespace additions are ENC_DISPLAY['virchow2_fp32'] = 'Virchow2' and,
for Run B (and B2) only, the single wrapper named in RUN_B_FEASIBILITY.md: `parse_xlsx_with_fp32_aliases`.

Subcommands (all refuse non-empty output directories; no LSF submission happens here):
  internals   hash-checked namespace load + structural facts (cheap; login node)
  eval-spec   M4.2 evaluator spec/launch/job files + leaf diff + line diff
  pins        write ROOT/<RUN_STAGE>/pins_v2main_001.json (after M3/M4.2 outputs exist)
  run-a       M4.3 Run A + bit-identity (A4) + evaluator cross-check (U-1)
  run-b       M4.3 Run B under option (ii) (B1-B4, A3 binding, A8, A10, U-4); requires the independent signed
              feasibility note
"""
import argparse
import copy
import csv
import difflib
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import types
import uuid

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

REPO = '/PROSPER_WORKSPACE/prosper_revision'
OUT = os.path.join(REPO, 'prosper_revision_output')
ROOT = os.path.join(OUT, 'r2_1_virchow2_fp32_main_20261013')
RUN_STAGE = 'run_003'  # mirror of submit.RUN_STAGE (AMENDMENT_004 / ADDENDUM_001 / ADDENDUM_003): default --run-dir
PS = '/PROSPER_WORKSPACE/prosper-search'
COMET = '/PROSPER_WORKSPACE/comet-feature-gen'
PS_HEAD = 'fd50373ce2176cd77c04f2c932f963f6e5e7dcd1'
RR5_PATH = os.path.join(REPO, 'scripts', 'r2_1_main_grid_plurality.py')
RR5_SHA = '4dac7f03b0f30fd79225658ecddf9c471792bcbedf5cd82fbce831436ef47e86'
RUN001 = os.path.join(OUT, 'r2_1_main_grid_plurality_20261012', 'run_001')
EVAL_TEMPLATE_SPEC = os.path.join(OUT, 'r2_1_main_grid_comparators_20261011', 'main_virchow2_spec.json')
EVAL_TEMPLATE_SPEC_SHA = '659a435843293767cf0e3c2fb2d8416e6a72046ca16aa88d7f38bc421570ae0b'
EVAL_TEMPLATE_JOB = os.path.join(OUT, 'r2_1_main_grid_comparators_20261011', 'main_virchow2_job.lsf')
EVAL_TEMPLATE_LAUNCH = os.path.join(OUT, 'r2_1_main_grid_comparators_20261011', 'main_virchow2_launch.sha256')
DRAWS_SHA = 'f81586230703cc578752a9b0a9363ddd4219f9cee28287b164b4115eb44a60a2'          # #46
POPULATIONS_SHA = '802b05aa6d84525d78500fe46bec9826eec78527d7fd06fb20ac45cebeb2ea5d'    # #45
EVAL_RUN_ID = 'r2_1_virchow2_fp32_main_20261013_eval001'
FROZEN_ROOT_NAMES = ('r2_1_main_grid_plurality_20261012', 'r2_1_main_grid_comparators_20261011',
                     'r2_1_virchow2_fp32_20261006', 'contract_candidate_')
FP32 = 'virchow2_fp32'
PUB = 'virchow2'
WRAPPER_NAME = 'parse_xlsx_with_fp32_aliases'
N_RR5_GATES = 13
PER_CELL_FILES = ('table_s3_revised.csv', 'hit_precision.csv', 'per_class.csv', 'tie_census.csv',
                  'tie_sensitivity.csv', 'per_query_predictions.csv.gz', 'aggregation_contrasts.csv',
                  'rule_contrasts.csv')
B1_FILES = ('table_s3_revised.csv', 'hit_precision.csv', 'per_class.csv', 'tie_census.csv', 'tie_sensitivity.csv',
            'rule_contrasts.csv', 'per_query_predictions.csv.gz')
A4_FILTERED = ('winners.csv', 'fig2b_matrix.csv')
A4_WHOLE = ('fig2a_panel.csv', 'fig2c_groups.csv', 'aggregation_contrasts.csv', 'titan_contrast.csv',
            'main_config_values.json')
B2_FILES = ('fig2a_panel.csv', 'fig2b_matrix.csv', 'fig2c_groups.csv', 'winners.csv', 'aggregation_contrasts.csv',
            'titan_contrast.csv', 'main_config_values.json')
NOT_GATED_JSON = ('lineage.json', 'input_bindings.json', 'COMPLETE.json', 'run_resources.json')
CSV_PRODUCTS = ('table_s3_revised.csv', 'hit_precision.csv', 'per_class.csv', 'tie_census.csv',
                'tie_sensitivity.csv', 'rule_contrasts.csv', 'aggregation_contrasts.csv', 'titan_contrast.csv',
                'per_query_predictions.csv.gz', 'fig2a_panel.csv', 'fig2b_matrix.csv', 'fig2c_groups.csv',
                'winners.csv')
FINAL_PRODUCTS = CSV_PRODUCTS + ('main_config_values.json',)
ID_COLS = ('cell_id', 'cell', 'pipeline', 'winner_cell', 'argmax_cell')      # A10 / U-4 identifier columns
A10_PER_CELL_ROW_TYPE = {'winners.csv': 'argmax_k', 'fig2b_matrix.csv': 'cell'}
FIG2C_FP32_LABEL = 'published Virchow2 matrix (submitted, mixed float16/float32 bank)'
# A8 as amended by the analyst (LEAD_IMPL_DECISIONS_001, F3): both RR-5 range_across_encoders scopes are relabelled.
A8_SCOPE_MAP = {'published_8': 'published_7+virchow2_fp32',
                'published_8_plus_h_optimus_1': 'published_7+virchow2_fp32_plus_h_optimus_1'}
A8_NOTE = 'expected False: fp32 replacement, not a reproduction gate'
SUBMITTED_FILL_COLS = ('submitted_value', 'submitted_value_from_published_matrix', 'submitted_ci_low',
                       'submitted_ci_high', 'submitted_ci_method')
SUBMITTED_SOURCE_COL = 'submitted_source'
SUBMITTED_SOURCE_FMT = 'virchow2/{m} (published matrix, submitted lineage)'
IMPORT_CLOSURE_PATTERNS = ('prosper/__init__.py', 'prosper/config.py', 'prosper/pipeline.py', 'prosper/embeddings',
                           'prosper/evaluation/__init__.py', 'prosper/evaluation/slide_headers.py',
                           'prosper/evaluation/metrics.py', 'prosper/evaluation/paths.py',
                           'prosper/evaluation/resample.py', 'prosper/evaluation/cli.py',
                           'prosper/evaluation/slide_eval.py', 'prosper/evaluation/metrics_cli.py')


class AdapterError(Exception):
    """A STOP condition of the adapter (never retried automatically)."""


# ----------------------------------------------------------------------------------------------
# hashing / IO helpers
# ----------------------------------------------------------------------------------------------

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def file_binding(p):
    return {'path': os.path.abspath(p), 'sha256': sha256_file(p), 'size': os.path.getsize(p)}


def canonical_json(obj):
    return (json.dumps(obj, indent=1, sort_keys=True) + '\n').encode()


def write_new(path, data):
    """Write a new file; never overwrite (FAILED/COMPLETE/products are immutable)."""
    if os.path.exists(path):
        raise AdapterError(f'refusing to overwrite {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def assert_writable_dir(path, frozen_roots=FROZEN_ROOT_NAMES, must_be_empty=True):
    ap = os.path.abspath(path)
    if ap.startswith('/tmp') or ap.startswith('/var/tmp'):
        raise AdapterError(f'{ap}: /tmp is forbidden')
    parts = ap.split(os.sep)
    for p in parts:
        for fr in frozen_roots:
            if p == fr or (fr.endswith('_') and p.startswith(fr)):
                raise AdapterError(f'{ap}: inside frozen root {p}')
    if must_be_empty and os.path.exists(ap) and (not os.path.isdir(ap) or os.listdir(ap)):
        raise AdapterError(f'{ap} exists and is not empty')
    return ap


def gz_decompress(b):
    return gzip.decompress(b)


def gz_compress_like_rr5(raw):
    """Same gzip parameters as RR-5 per_query_rows (mtime 0, empty filename, default level)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as gz:
        gz.write(raw)
    return buf.getvalue()


def read_product(d, name):
    with open(os.path.join(d, name), 'rb') as f:
        b = f.read()
    return gz_decompress(b) if name.endswith('.gz') else b


# ----------------------------------------------------------------------------------------------
# CSV line utilities (byte-exact row selection; no re-serialisation of untouched rows)
# ----------------------------------------------------------------------------------------------

def split_lines(data):
    if b'\r' in data:
        raise AdapterError('unexpected carriage return in a product')
    if not data.endswith(b'\n'):
        raise AdapterError('product does not end with a newline')
    lines = data[:-1].split(b'\n')
    return lines[0], lines[1:]


def parse_fields(line):
    rows = list(csv.reader([line.decode('utf-8')]))
    if len(rows) != 1:
        raise AdapterError('a CSV line did not parse to exactly one record')
    return rows[0]


def format_fields(fields):
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n', quoting=csv.QUOTE_MINIMAL).writerow(fields)
    return buf.getvalue()[:-1].encode('utf-8')


def table(data):
    """(header bytes, header fields, [(line bytes, fields)]); each line must have the header's width."""
    head, lines = split_lines(data)
    hf = parse_fields(head)
    out = []
    for ln in lines:
        f = parse_fields(ln)
        if len(f) != len(hf):
            raise AdapterError(f'CSV line width {len(f)} != header width {len(hf)} (embedded newline?)')
        out.append((ln, f))
    return head, hf, out


def join_lines(head, lines):
    return head + b'\n' + b''.join(ln + b'\n' for ln in lines)


def id_positions(hf):
    return [hf.index(c) for c in ID_COLS if c in hf]


def rows_for_cell(data, cid):
    head, hf, rows = table(data)
    col = hf.index('cell') if 'cell' in hf and 'cell_id' not in hf else hf.index('cell_id')
    return head, [ln for ln, f in rows if f[col] == cid]


# ----------------------------------------------------------------------------------------------
# hash-checked RR-5 namespace
# ----------------------------------------------------------------------------------------------

def load_rr5(path=RR5_PATH, expected_sha=RR5_SHA):
    """Read the bytes, assert sha256, compile and execute them in a fresh module namespace."""
    with open(path, 'rb') as f:
        src = f.read()
    got = sha256_bytes(src)
    if got != expected_sha:
        raise AdapterError(f'RR-5 sha256 {got} != {expected_sha}')
    name = f'rr5_ns_{uuid.uuid4().hex}'
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, 'exec'), mod.__dict__)  # noqa: S102 - hash-checked frozen source
    mod.__rr5_sha256__ = got
    return mod


def add_fp32_display(ns):
    """Namespace addition (i). XLSX_EMB was computed at import, so it keeps Virchow2 -> virchow2."""
    if FP32 in ns.ENC_DISPLAY:
        raise AdapterError('ENC_DISPLAY already has virchow2_fp32')
    ns.ENC_DISPLAY[FP32] = 'Virchow2'
    if ns.XLSX_EMB.get('Virchow2') != PUB or FP32 in ns.XLSX_EMB.values():
        raise AdapterError('XLSX_EMB changed by the display addition')


def install_alias_wrapper(ns, record):
    """The single named wrapper (RUN_B_FEASIBILITY.md, option (ii)). Calls the original parse_xlsx with exactly
    run()'s arguments and only adds the ('virchow2_fp32/<m>', k) keys as the identical row objects of
    ('virchow2/<m>', k) for the fp32 cells present in the spec (16 under rb_spec, 0 under the 38-cell spec)."""
    orig = ns.parse_xlsx
    if getattr(orig, '__name__', '') == WRAPPER_NAME:
        raise AdapterError('wrapper already installed')

    def parse_xlsx_with_fp32_aliases(spec, inp, popd, hist):
        xl = orig(spec, inp, popd, hist)
        rows = xl['rows']
        n = 0
        for c in spec['cells']:
            if c['enc'] != FP32:
                continue
            for k in ns.K_VALUES:
                src, dst = (f'{PUB}/{c["method"]}', k), (c['cell_id'], k)
                if dst in rows:
                    raise AdapterError(f'alias target {dst} already present')
                rows[dst] = rows[src]
                if rows[dst] is not rows[src]:
                    raise AdapterError(f'alias {dst} is not the identical object of {src}')
                n += 1
        record['alias_count'] = n
        record['aliases'] = sorted(f'{c}@{k}' for c, k in rows if c.startswith(FP32 + '/'))
        record['calls'] = record.get('calls', 0) + 1
        return xl

    ns.parse_xlsx = parse_xlsx_with_fp32_aliases
    record['wrapper'] = WRAPPER_NAME
    return parse_xlsx_with_fp32_aliases


# ----------------------------------------------------------------------------------------------
# specs and pins
# ----------------------------------------------------------------------------------------------

def leaves(obj, prefix=''):
    if isinstance(obj, dict):
        out = {}
        for k in obj:
            out.update(leaves(obj[k], f'{prefix}.{k}' if prefix else str(k)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(leaves(v, f'{prefix}[{i}]'))
        if not obj:
            out[prefix] = []
        return out
    return {prefix: obj}


def leaf_diff(a, b):
    la, lb = leaves(a), leaves(b)
    keys = sorted(set(la) | set(lb))
    return [{'leaf': k, 'old': la.get(k, '<absent>'), 'new': lb.get(k, '<absent>')} for k in keys
            if la.get(k, '<absent>') != lb.get(k, '<absent>') or (k in la) != (k in lb)]


def fp32_cell(ns, m, path, sha):
    """Exact Major-4 fp32 cell dict (Run A and, under option (ii), Run B: xlsx_key stays None)."""
    return dict(cell_id=f'{FP32}/{m}', enc=FP32, method=m, display_encoder='Virchow2',
                aggregation=ns.AGG_DISPLAY[m], lineage='virchow2_fp32_replacement', submitted_lineage=False,
                reconcile=False, path=path, sha256=sha, hist_prefix=None, xlsx_key=None)


def build_full_spec(ns, base_spec, fp32_paths, eval_dir, pins_path, pins_sha):
    spec = copy.deepcopy(base_spec)
    n0 = len(spec['cells'])
    for m in ns.METHODS:
        p = fp32_paths[m]
        spec['cells'].append(fp32_cell(ns, m, p, sha256_file(p)))
    spec['qm_encoders'] = list(spec['qm_encoders']) + [FP32]
    spec['paths']['pins_file'] = pins_path
    spec['paths'][f'summary_{FP32}'] = os.path.join(eval_dir, 'summary.json')
    spec['paths'][f'qm_{FP32}'] = os.path.join(eval_dir, 'query_metrics.csv.gz')
    spec['pins']['pins_file'] = pins_sha
    spec['pins']['qm'][FP32] = sha256_file(spec['paths'][f'qm_{FP32}'])
    spec['pins']['summary'][FP32] = sha256_file(spec['paths'][f'summary_{FP32}'])
    allowed = {'paths.pins_file', f'paths.summary_{FP32}', f'paths.qm_{FP32}', 'pins.pins_file',
               f'pins.qm.{FP32}', f'pins.summary.{FP32}', f'qm_encoders[{len(base_spec["qm_encoders"])}]'}
    for d in leaf_diff(base_spec, spec):
        lf = d['leaf']
        if lf in allowed:
            continue
        if lf.startswith('cells[') and int(lf[6:lf.index(']')]) >= n0:
            continue
        raise AdapterError(f'full_spec changes an unlisted key: {lf}')
    return spec


def import_closure(ps=PS, head=PS_HEAD, observed=None, run=subprocess.run):
    """m9: closure file list at HEAD (+ the empty-diff assertion) and an optional observed {module: sha256}."""
    ls = run(['git', '-C', ps, 'ls-files', '--', *IMPORT_CLOSURE_PATTERNS], capture_output=True, text=True,
             check=True).stdout.split()
    diff = run(['git', '-C', ps, 'diff', '--quiet', head, '--', *ls])
    if diff.returncode != 0:
        raise AdapterError('import-closure diff against the pinned HEAD is not empty')
    return {'head': head, 'patterns': list(IMPORT_CLOSURE_PATTERNS),
            'files': {f: sha256_file(os.path.join(ps, f)) for f in ls},
            'observed_sys_modules': observed or {}}


def build_pins(base_pins_bytes, cells, eval_dir, closure):
    pf = json.loads(base_pins_bytes)
    fp = [c for c in cells if c['enc'] == FP32]
    if len(fp) != 4:
        raise AdapterError(f'{len(fp)} fp32 cells, expected 4')
    for c in fp:
        if c['cell_id'] in pf['matrices']:
            raise AdapterError(f'pins already has {c["cell_id"]}')
        pf['matrices'][c['cell_id']] = {'path': c['path'], 'sha256': c['sha256']}
    pf['packages'][FP32] = {
        'draws_sha256': sha256_file(os.path.join(eval_dir, 'patient_draws.npy')),
        'populations_sha256': sha256_file(os.path.join(eval_dir, 'populations.csv')),
        'query_metrics_sha256': sha256_file(os.path.join(eval_dir, 'query_metrics.csv.gz')),
        'summary_sha256': sha256_file(os.path.join(eval_dir, 'summary.json'))}
    if 'import_closure' in pf:
        raise AdapterError('base pins already carries import_closure')
    pf['import_closure'] = closure
    return canonical_json(pf)


def pre_run_asserts(spec, pins_bytes):
    """[m1] set equality of pins matrices with the cells; fp32 pins sha equals the cell sha. Pins entries are
    {path, sha256} dicts (the form cross_check_pins_file reads), so the sha256 member is compared."""
    pins = json.loads(pins_bytes)
    cells = {c['cell_id']: c for c in spec['cells']}
    if set(pins['matrices']) != set(cells):
        raise AdapterError('pins matrices != cells')
    for cid, c in cells.items():
        if c['enc'] == FP32:
            e = pins['matrices'][cid]
            if not isinstance(e, dict) or e.get('sha256') != c['sha256']:
                raise AdapterError(f'pins sha for {cid} != cell sha')
            if sha256_file(c['path']) != c['sha256']:
                raise AdapterError(f'{cid}: file sha != cell sha')
    if sha256_bytes(pins_bytes) != spec['pins']['pins_file']:
        raise AdapterError('pins file sha != spec pins.pins_file')
    hj = pins.get('historical_json', {})
    if len(hj) != spec['n_hist_rows']:
        raise AdapterError('historical_json size != n_hist_rows')
    return {'n_matrices': len(pins['matrices']), 'n_fp32': 4, 'historical_json': len(hj)}


def make_rb_spec(full_spec):
    """Option (ii): only contested_encoders virchow2->virchow2_fp32 (same position) and fig2c.pipelines
    virchow2/yottixel -> virchow2_fp32/yottixel. Cells, xlsx_key, qm_encoders, pins: unchanged."""
    rb = copy.deepcopy(full_spec)
    ce = rb['contested_encoders']
    if ce.count(PUB) != 1 or FP32 in ce:
        raise AdapterError('contested_encoders does not hold exactly one virchow2')
    ce[ce.index(PUB)] = FP32
    pl = rb['fig2c']['pipelines']
    if pl.count(f'{PUB}/yottixel') != 1:
        raise AdapterError('fig2c.pipelines does not hold exactly one virchow2/yottixel')
    pl[pl.index(f'{PUB}/yottixel')] = f'{FP32}/yottixel'
    d = leaf_diff(full_spec, rb)
    if len(d) != 2 or not all(x['leaf'].startswith(('contested_encoders[', 'fig2c.pipelines[')) for x in d):
        raise AdapterError(f'rb_spec differs from full_spec in unexpected leaves: {d}')
    return rb, d


# ----------------------------------------------------------------------------------------------
# running RR-5
# ----------------------------------------------------------------------------------------------

def run_rr5(ns, spec, out_dir, spec_path, n_reps, extra=None):
    """RR-5's own main() on {'full_spec': spec}; returns (rc, COMPLETE.json or FAILED.json)."""
    rec = {'full_spec': spec, 'full_spec_sha256': sha256_bytes(canonical_json(spec))}
    rec.update(extra or {})
    write_new(spec_path, canonical_json(rec))
    rc = ns.main(['--out-dir', out_dir, '--spec', spec_path, '--n-reps', str(int(n_reps))])
    status = None
    for name in ('COMPLETE.json', 'FAILED.json'):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            status = json.load(open(p))
            status['_file'] = name
    return rc, status


def require_complete(rc, status, label, n_gates=N_RR5_GATES):
    if rc != 0 or not status or status.get('_file') != 'COMPLETE.json' or status.get('status') != 'COMPLETE':
        raise AdapterError(f'{label}: RR-5 did not complete (rc={rc}, {status and status.get("_file")}, '
                           f'{status and (status.get("gate"), status.get("reason"))})')
    if len(status.get('gates_passed', [])) != n_gates:
        raise AdapterError(f'{label}: {len(status.get("gates_passed", []))} gates passed, expected {n_gates}')
    return status['gates_passed']


def recompute_cell_stats(ns, spec, cell_ids, bind_dir, n_reps, keep=None):
    """RR-5's own call path (gate_binding -> gate_draws -> load_cell_matrix -> rank_cell -> predict_cell ->
    cell_statistics) for the named cells, from one load of the draws and one population ordering (A5/U-1/U-5).
    The same `W` object is passed to cell_statistics for every cell (F5; its sha256(W.tobytes()) is in ctx).
    Returns {cid: st} and the context (popd, labels_order, n_reps, W_sha256). No statistic is reimplemented here.
    If `keep` is a dict, keep[cid] receives RR-5's own rank_cell (top, flags, census) for M5.3."""
    os.makedirs(bind_dir, exist_ok=True)
    ns.check_scope(spec)
    inp = ns.Inputs(spec)
    popd, hist, xl, (plurality, legacy_generous, _v) = ns.gate_binding(spec, inp, bind_dir)
    draws, pcol, _P = ns.gate_draws(spec, inp, popd)
    if sha256_bytes(inp.bytes['draws']) != DRAWS_SHA and spec.get('n_reps') == 5000:
        raise AdapterError('draws sha != #46')
    W = draws[:n_reps][:, pcol].astype(np.float64)
    del draws
    labels_order, y = popd['labels_order'], popd['y']
    T = ns.onehot(y, len(labels_order))
    S_rep = W @ T
    S_pt = T.sum(axis=0)
    cells = {c['cell_id']: c for c in spec['cells']}
    stats = {}
    for cid in cell_ids:
        X, same, _ax = ns.load_cell_matrix(spec, inp, cells[cid], popd)
        top, _fl, _cen = ns.rank_cell(X, same, y)
        del X
        if keep is not None:
            keep[cid] = {'top': top, 'flags': _fl, 'census': _cen}
        preds = ns.predict_cell(top, y, labels_order, plurality, legacy_generous)
        stats[cid] = ns.cell_statistics(W, T, S_rep, S_pt, y, preds, len(labels_order))
    ctx = {'popd': popd, 'labels_order': labels_order, 'n_reps': int(W.shape[0]),
           'draws_sha256': sha256_bytes(inp.bytes['draws']), 'bindings': inp.bindings,
           'W_sha256': sha256_bytes(np.ascontiguousarray(W).tobytes()), 'W_shape': list(W.shape),
           'W_dtype': str(W.dtype)}
    return stats, ctx


def fmt6(ns, x):
    """RR-5's writer format for a float (pandas round(DP) then '%.6f'); NaN -> ''."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ''
    b = ns._csv_bytes(pd.DataFrame({'v': [float(x)]}))
    return b.decode().split('\n')[1]


# ----------------------------------------------------------------------------------------------
# Run A: bit identity (specification M4.3 + A4) and evaluator cross-check (U-1)
# ----------------------------------------------------------------------------------------------

def json_key_diff(a, b):
    ka, kb = set(a), set(b)
    return {'added': sorted(kb - ka), 'removed': sorted(ka - kb),
            'changed': sorted(k for k in ka & kb if json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True))}


def bit_identity_report(run_a, run001, orig_cells, fp32_cells):
    rep = {'per_cell': {}, 'a4_filtered': {}, 'a4_whole_file': {}, 'reconciliation': {}, 'not_gated': {},
           'fp32_order_sensitive_10': {}, 'failures': []}
    fail = rep['failures']
    for name in PER_CELL_FILES:
        a, b = read_product(run_a, name), read_product(run001, name)
        ha, _, rows_a = table(a)
        hb, _, _ = table(b)
        if ha != hb:
            fail.append(f'{name}: header differs')
        n_checked, n_rows = 0, 0
        for cid in orig_cells:
            _, ra = rows_for_cell(a, cid)
            _, rb = rows_for_cell(b, cid)
            if ra != rb:
                fail.append(f'{name}: rows of {cid} differ ({len(ra)} vs {len(rb)})')
            if not ra and name != 'aggregation_contrasts.csv':
                fail.append(f'{name}: no Run A rows for {cid}')
            n_checked += 1
            n_rows += len(ra)
        fp_rows = 0
        for cid in fp32_cells:
            _, ra = rows_for_cell(a, cid)
            if not ra and name != 'aggregation_contrasts.csv':
                fail.append(f'{name}: no Run A rows for {cid}')
            fp_rows += len(ra)
        if n_rows + fp_rows != len(rows_a):
            fail.append(f'{name}: {len(rows_a) - n_rows - fp_rows} Run A rows belong to no declared cell')
        rep['per_cell'][name] = {'cells_checked': n_checked, 'rows_identical': n_rows, 'fp32_rows': fp_rows}
    for name in A4_FILTERED:
        a, b = read_product(run_a, name), read_product(run001, name)
        head, lines = split_lines(a)
        kept = [ln for ln in lines if f'{FP32}/'.encode() not in ln]
        ok = join_lines(head, kept) == b
        rep['a4_filtered'][name] = {'identical_after_removing_fp32_rows': ok, 'fp32_rows_removed':
                                    len(lines) - len(kept)}
        if not ok:
            fail.append(f'{name}: not byte-identical to run_001 after removing virchow2_fp32/ rows (A4)')
    for name in A4_WHOLE:
        ok = read_product(run_a, name) == read_product(run001, name)
        rep['a4_whole_file'][name] = ok
        if not ok:
            fail.append(f'{name}: not whole-file byte-identical to run_001 (A4)')
    ra = json.load(open(os.path.join(run_a, 'reconciliation.json')))
    rb = json.load(open(os.path.join(run001, 'reconciliation.json')))
    for k in sorted(set(ra) | set(rb)):
        if k == 'hit_gate':
            continue
        same = json.dumps(ra.get(k), sort_keys=True) == json.dumps(rb.get(k), sort_keys=True)
        rep['reconciliation'][k] = same
        if not same:
            fail.append(f'reconciliation.json[{k}] differs')
    hg_a, hg_b = ra.get('hit_gate', {}), rb.get('hit_gate', {})
    for e in sorted(set(hg_a) | set(hg_b)):
        if e == FP32:
            rep['reconciliation'][f'hit_gate.{e}'] = 'added'
            continue
        same = json.dumps(hg_a.get(e), sort_keys=True) == json.dumps(hg_b.get(e), sort_keys=True)
        rep['reconciliation'][f'hit_gate.{e}'] = same
        if not same:
            fail.append(f'reconciliation.json hit_gate[{e}] differs')
    if FP32 not in hg_a:
        fail.append('reconciliation.json hit_gate lacks virchow2_fp32')
    else:
        rep['hit_gate_fp32'] = hg_a[FP32]
    for name in NOT_GATED_JSON:
        try:
            rep['not_gated'][name] = json_key_diff(json.load(open(os.path.join(run001, name))),
                                                   json.load(open(os.path.join(run_a, name))))
        except (OSError, ValueError) as e:
            rep['not_gated'][name] = f'unreadable: {e}'
    _, hf, rows = table(read_product(run_a, 'tie_census.csv'))
    ci, oi = hf.index('cell_id'), hf.index('order_sensitive_10')
    tc = {f[ci]: f[oi] for _, f in rows}
    for cid in fp32_cells:
        pub = PUB + cid[len(FP32):]
        rep['fp32_order_sensitive_10'][cid] = {'fp32': tc.get(cid), 'published': tc.get(pub),
                                               'note': 'descriptive; no fp32 tie-census anchor (Q3)'}
    rep['pass'] = not fail
    return rep


def evaluator_crosscheck(ns, stats, estimates_csv, run_a, methods, k_values, tol=1e-12):
    """U-1 (a): evaluator macro hit/precision points vs RR-5 unrounded in-memory points, |d| <= 1e-12 (STOP);
    plus the descriptive 6-decimal string equality with hit_precision.csv."""
    est = pd.read_csv(estimates_csv)
    est = est[(est['row_kind'] == 'estimate') & (est['level'] == 'macro') & est['endpoint'].isin(['hit', 'precision'])]
    _, hf, hp = table(read_product(run_a, 'hit_precision.csv'))
    ix = {(f[hf.index('cell_id')], f[hf.index('k')], f[hf.index('tie_order')], f[hf.index('metric')]):
          f[hf.index('value')] for _, f in hp}
    rows, bad = [], []
    for m in methods:
        cid = f'{FP32}/{m}'
        for k in k_values:
            for ep, metric in (('hit', 'hit_macro'), ('precision', 'precision_macro')):
                s = est[(est['method'] == m) & (est['endpoint'] == ep) & (est['k'] == k)]
                if len(s) != 1:
                    bad.append(f'{cid} k={k} {ep}: {len(s)} evaluator rows')
                    continue
                ev = float(s['point'].iloc[0])
                rr = stats[cid][('canonical', k, 'hp')][metric].point
                d = abs(ev - rr)
                six = ix.get((cid, str(k), 'canonical', metric))
                row = {'cell_id': cid, 'k': k, 'metric': metric, 'evaluator_point': repr(ev),
                       'rr5_unrounded_point': repr(rr), 'abs_diff': d, 'pass': d <= tol,
                       'hit_precision_csv_value': six, 'six_dp_string_equal': six == fmt6(ns, ev)}
                rows.append(row)
                if d > tol:
                    bad.append(f'{cid} k={k} {metric}: |d|={d:.3g} > {tol}')
    n_exp = len(methods) * len(k_values) * 2
    if len(rows) != n_exp:
        bad.append(f'{len(rows)} comparisons, expected {n_exp}')
    return {'tolerance': tol, 'rule': 'U-1 option (a): unrounded in-memory RR-5 points via the hash-checked call '
                                     'path; 6-dp string equality is descriptive',
            'n_compared': len(rows), 'rows': rows, 'failures': bad, 'pass': not bad}


# ----------------------------------------------------------------------------------------------
# Run B: B1, B2, A10, A8, Major-4 fill, U-4
# ----------------------------------------------------------------------------------------------

def b1_check(raw_b, run_a):
    res, bad = {}, []
    for name in B1_FILES:
        ok = read_product(raw_b, name) == read_product(run_a, name)
        res[name] = ok
        if not ok:
            bad.append(f'B1: {name} not whole-file byte-identical to Run A')
    return res, bad


def b2_check(b2_dir, run001):
    res, bad = {}, []
    for name in B2_FILES:
        ok = read_product(b2_dir, name) == read_product(run001, name)
        res[name] = ok
        if not ok:
            bad.append(f'B2: {name} differs from run_001')
    return res, bad


def _is_pub_id(f, pos):
    return any(f[i].startswith(f'{PUB}/') for i in pos)


def a10_drop(raw_b, run_a):
    """A10: drop from each raw Run B product exactly the rows whose identifier (ID_COLS) starts with 'virchow2/'.
    The dropped rows must equal, byte for byte and in order, the Run A rows for those four cells:
      - B1 per-cell files: every Run A row carrying a virchow2/ identifier;
      - winners.csv / fig2b_matrix.csv: the Run A per-cell rows of row_type argmax_k / cell carrying one;
      - every other product (slot- or pipeline-level only): nothing may be dropped.
    Retained rows are the raw Run B lines, unchanged and in raw order."""
    out, counts, bad = {}, {}, []
    for name in CSV_PRODUCTS:
        b = read_product(raw_b, name)
        head, hf, rows = table(b)
        pos = id_positions(hf)
        dropped = [ln for ln, f in rows if _is_pub_id(f, pos)]
        kept = [ln for ln, f in rows if not _is_pub_id(f, pos)]
        a = read_product(run_a, name)
        ha, hfa, rows_a = table(a)
        posa = id_positions(hfa)
        a_sel = [(ln, f) for ln, f in rows_a if _is_pub_id(f, posa)]
        if name in B1_FILES:
            a_cmp = [ln for ln, _ in a_sel]
            rule = 'all Run A virchow2/ rows'
        elif name in A10_PER_CELL_ROW_TYPE:
            rt = A10_PER_CELL_ROW_TYPE[name]
            a_cmp = [ln for ln, f in a_sel if f[hfa.index('row_type')] == rt]
            rule = f'Run A virchow2/ rows of row_type {rt}'
        else:
            a_cmp = []
            rule = 'no row may be dropped'
        if dropped != a_cmp or ha != head:
            bad.append(f'A10: {name}: {len(dropped)} dropped rows != {len(a_cmp)} Run A rows ({rule})')
        if len(kept) + len(dropped) != len(rows):
            bad.append(f'A10: {name}: row accounting')
        out[name] = (head, kept)
        counts[name] = {'raw_rows': len(rows), 'dropped': len(dropped), 'retained': len(kept),
                        'run_a_rows_compared': len(a_cmp), 'comparison': rule}
    return out, counts, bad


def _edit_column(head, lines, col, pred, new_value):
    hf = parse_fields(head)
    j = hf.index(col)
    out, n = [], 0
    for ln in lines:
        f = parse_fields(ln)
        if format_fields(f) != ln:
            raise AdapterError('CSV round-trip of a raw line is not byte-identical; refusing to edit')
        if pred(hf, f):
            f[j] = new_value
            ln = format_fields(f)
            n += 1
        out.append(ln)
    return out, n


def a8_and_major4_edits(kept, run_a, methods):
    """The only post-call edits: fig2b range scope (A8), fig2c submitted_from (Major-4), table_s3 submitted_*
    fill + submitted_source column (Major-4)."""
    edits = {}
    head, lines = kept['fig2b_matrix.csv']
    n = 0
    per_scope = {}
    for frm, to in A8_SCOPE_MAP.items():
        lines, n_s = _edit_column(head, lines, 'scope',
                                  lambda hf, f, frm=frm: f[hf.index('row_type')].startswith('range_') and
                                  f[hf.index('scope')] == frm, to)
        per_scope[frm] = n_s
        n += n_s
    hf_b = parse_fields(head)
    left = [ln for ln in lines if parse_fields(ln)[hf_b.index('scope')] in A8_SCOPE_MAP]
    if left:
        raise AdapterError(f'A8: {len(left)} fig2b rows still carry a published_8* scope after relabelling')
    kept['fig2b_matrix.csv'] = (head, lines)
    edits['fig2b_scope_relabelled'] = n
    edits['fig2b_scope_relabelled_by_scope'] = per_scope
    head, lines = kept['fig2c_groups.csv']
    lines, n = _edit_column(head, lines, 'submitted_from',
                            lambda hf, f: f[hf.index('pipeline')] == f'{FP32}/yottixel', FIG2C_FP32_LABEL)
    kept['fig2c_groups.csv'] = (head, lines)
    edits['fig2c_submitted_from_relabelled'] = n
    _, hfa, rows_a = table(read_product(run_a, 'table_s3_revised.csv'))
    key = ('k', 'rule', 'tie_order', 'metric')
    src = {}
    for _, f in rows_a:
        if f[hfa.index('cell_id')].startswith(f'{PUB}/'):
            src[(f[hfa.index('cell_id')],) + tuple(f[hfa.index(c)] for c in key)] = \
                {c: f[hfa.index(c)] for c in SUBMITTED_FILL_COLS}
    head, lines = kept['table_s3_revised.csv']
    hf = parse_fields(head)
    if SUBMITTED_SOURCE_COL in hf:
        raise AdapterError('table_s3 already has submitted_source')
    new_lines, n_fill = [], 0
    for ln in lines:
        f = parse_fields(ln)
        if format_fields(f) != ln:
            raise AdapterError('CSV round-trip of a raw table_s3 line is not byte-identical')
        cid = f[hf.index('cell_id')]
        extra = ''
        if cid.startswith(f'{FP32}/'):
            m = cid.split('/', 1)[1]
            s = src.get((f'{PUB}/{m}',) + tuple(f[hf.index(c)] for c in key))
            if s is None:
                raise AdapterError(f'no Run A virchow2/{m} row for {cid} {[f[hf.index(c)] for c in key]}')
            for c in SUBMITTED_FILL_COLS:
                f[hf.index(c)] = s[c]
            extra = SUBMITTED_SOURCE_FMT.format(m=m)
            n_fill += 1
        new_lines.append(format_fields(f + [extra]))
    kept['table_s3_revised.csv'] = (format_fields(hf + [SUBMITTED_SOURCE_COL]), new_lines)
    edits['table_s3_fp32_rows_filled'] = n_fill
    edits['table_s3_submitted_source_column_added'] = True
    return kept, edits


def u4_assert(final_dir):
    """U-4 on final replacement product files only."""
    bad = []
    for name in CSV_PRODUCTS:
        _, hf, rows = table(read_product(final_dir, name))
        pos = id_positions(hf)
        allowed = {i for i, c in enumerate(hf) if c.startswith('submitted_') or c == 'lineage'}
        de = hf.index('display_encoder') if 'display_encoder' in hf else None
        for ln, f in rows:
            if de is not None and f[de] == 'Virchow2':
                for i in pos:
                    if f[i] and not f[i].startswith(f'{FP32}/'):
                        bad.append(f'{name}: Virchow2 row with identifier {f[i]!r}')
            for i, v in enumerate(f):
                if f'{PUB}/' in v and i not in allowed:
                    bad.append(f'{name}: virchow2/ in column {hf[i]!r}')
    mcv = read_product(final_dir, 'main_config_values.json').decode()
    if f'{PUB}/' in mcv:
        bad.append('main_config_values.json contains virchow2/')
    return bad[:200]


# ----------------------------------------------------------------------------------------------
# M4.2 evaluator spec / job
# ----------------------------------------------------------------------------------------------

def eval_spec(template_bytes, matrices, ranks, run_id=EVAL_RUN_ID):
    t = json.loads(template_bytes)
    reser = (json.dumps(t, indent=2, sort_keys=True) + '\n').encode()
    if sha256_bytes(reser) != sha256_bytes(template_bytes):
        raise AdapterError('json.dumps(indent=2, sort_keys=True) does not reproduce the template bytes')
    s = copy.deepcopy(t)
    s['run_id'] = run_id
    set_leaves = ['run_id']
    for m in sorted(matrices):
        for kind, src in (('matrix', matrices[m]), ('ranks', ranks[m])):
            b = file_binding(src)
            for f in ('path', 'sha256', 'size'):
                s['sources'][f'{kind}_{m}'][f] = b[f]
                set_leaves.append(f'sources.{kind}_{m}.{f}')
    out = (json.dumps(s, indent=2, sort_keys=True) + '\n').encode()
    d = leaf_diff(t, s)
    changed = {x['leaf'] for x in d}
    extra = changed - set(set_leaves)
    lt, ls = leaves(t), leaves(s)
    listed = [{'leaf': k, 'old': lt[k], 'new': ls[k], 'value_changed': lt[k] != ls[k]} for k in set_leaves]
    diff = {'template_sha256': sha256_bytes(template_bytes), 'serialisation': 'json.dumps(indent=2, sort_keys=True)+"\\n" '
            '(reproduces the template sha256)', 'leaves': listed, 'n_leaves': len(listed),
            'n_value_changed': sum(x['value_changed'] for x in listed), 'unexpected_changes': sorted(extra),
            'note': 'the 25 set leaves are always listed; value_changed is False only if a new value happens to '
                    'equal the template value (e.g. an identical file size)'}
    if extra or len(set_leaves) != 25 or any(k not in lt for k in set_leaves):
        raise AdapterError(f'eval spec diff outside the 25 leaves: {sorted(extra)}')
    return out, diff


def eval_launch(template_launch, template_spec_path, new_spec_path, new_spec_sha, old_sources, new_sources):
    """Template launch.sha256 with the spec, matrix and ranks lines rebound; all other lines unchanged."""
    repl = {template_spec_path: (new_spec_sha, new_spec_path)}
    for k, old in old_sources.items():
        repl[old['path']] = (new_sources[k]['sha256'], new_sources[k]['path'])
    out, n = [], 0
    for ln in template_launch.decode().splitlines():
        h, p = ln.split('  ', 1)
        if p in repl:
            h, p = repl[p]
            n += 1
        out.append(f'{h}  {p}')
    if n != len(repl):
        raise AdapterError(f'{n} launch lines rebound, expected {len(repl)}')
    return ('\n'.join(out) + '\n').encode()


def eval_job(template_job, queue, run_dir, spec_path, spec_sha, launch_path, launch_sha, job_name='r21_v2main_fp32_eval'):
    t = template_job.decode()
    tmpl_dir = os.path.dirname(EVAL_TEMPLATE_JOB)
    attempt = os.path.join(run_dir, 'eval_attempt_001')
    rules = [
        ('#BSUB -J ', f'#BSUB -J {job_name}'),
        ('#BSUB -q ', f'#BSUB -q {queue}'),
        ('#BSUB -oo ', f'#BSUB -oo {run_dir}/eval_job_%J.out'),
        ('#BSUB -eo ', f'#BSUB -eo {run_dir}/eval_job_%J.err'),
        ('export PROSPER_OUT=', f'export PROSPER_OUT="{run_dir}"'),
        ("printf '%s\\n' ", f"printf '%s\\n' '{launch_sha}  {launch_path}' | sha256sum --check --strict -"),
        ('sha256sum --check --strict "', f'sha256sum --check --strict "{launch_path}"'),
        ('test ! -e ', f'test ! -e "{attempt}"'),
        ('  --spec ', f'  --spec "{spec_path}" --spec-sha256 {spec_sha} \\'),
        ('  --out ', f'  --out "{attempt}" --out-root "{run_dir}"'),
    ]
    out, used = [], []
    for ln in t.split('\n'):
        new = ln
        for pre, rep in rules:
            if ln.startswith(pre):
                new = rep
                used.append(pre)
                break
        out.append(new)
    if sorted(used) != sorted(p for p, _ in rules):
        raise AdapterError(f'eval job template lines not matched exactly once: {used}')
    job = '\n'.join(out)
    if tmpl_dir in job:
        raise AdapterError('template directory still referenced in the new job')
    diff = ''.join(difflib.unified_diff(t.splitlines(True), job.splitlines(True), 'main_virchow2_job.lsf (#44)',
                                        'eval_job_v2main_001.lsf'))
    return job.encode(), diff.encode()


# ----------------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------------

def fp32_matrix_paths(run_dir, methods):
    return {m: os.path.join(run_dir, 'similarity_matrix', f'combined_similarity_matrix_{FP32}_{m}_no_same_patient.pkl')
            for m in methods}


def cmd_internals(args):
    ns = load_rr5(args.rr5, args.rr5_sha)
    add_fp32_display(ns)
    spec = ns.contract_spec(REPO, COMET, OUT)
    facts = {'rr5_sha256': ns.__rr5_sha256__, 'n_cells': len(spec['cells']), 'n_hist_rows': spec['n_hist_rows'],
             'contested_encoders': spec['contested_encoders'], 'qm_encoders': spec['qm_encoders'],
             'fig2c_pipelines': spec['fig2c']['pipelines'], 'ENC_DISPLAY_fp32': ns.ENC_DISPLAY[FP32],
             'XLSX_EMB_Virchow2': ns.XLSX_EMB['Virchow2'],
             'functions': sorted(n for n in ('parse_xlsx', 'gate_binding', 'cross_check_pins_file', 'declared_inputs',
                                             'gate_draws', 'fig2c_identification', 'reconcile', 'hit_gate', 'slots',
                                             'cell_statistics', 'contrast_stat', 'main', 'run')
                                 if callable(getattr(ns, n, None)))}
    print(json.dumps(facts, indent=1))
    return 0


def cmd_eval_spec(args):
    run_dir = os.path.abspath(args.run_dir)
    assert_writable_dir(run_dir, must_be_empty=False)
    tb = open(EVAL_TEMPLATE_SPEC, 'rb').read()
    if sha256_bytes(tb) != EVAL_TEMPLATE_SPEC_SHA:
        raise AdapterError('template spec sha mismatch')
    methods = ('median_pooling', 'ot_cosine', 'sampler', 'yottixel')
    mats = fp32_matrix_paths(run_dir, methods)
    ranks = {m: os.path.join(run_dir, 'c2_metric_suite', f'ranked_lists_{FP32}_{m}.csv.gz') for m in methods}
    spec_b, diff = eval_spec(tb, mats, ranks)
    sp = os.path.join(run_dir, 'eval_spec_v2main_001.json')
    write_new(sp, spec_b)
    write_new(os.path.join(run_dir, 'eval_spec_diff.json'), canonical_json(diff))
    t = json.loads(tb)
    new = json.loads(spec_b)
    old_src = {k: t['sources'][k] for k in t['sources'] if k.startswith(('matrix_', 'ranks_'))}
    new_src = {k: new['sources'][k] for k in old_src}
    lb = eval_launch(open(EVAL_TEMPLATE_LAUNCH, 'rb').read(), EVAL_TEMPLATE_SPEC, sp, sha256_bytes(spec_b),
                     old_src, new_src)
    lp = os.path.join(run_dir, 'eval_launch_v2main_001.sha256')
    write_new(lp, lb)
    job, jd = eval_job(open(EVAL_TEMPLATE_JOB, 'rb').read(), args.queue, run_dir, sp, sha256_bytes(spec_b), lp,
                       sha256_bytes(lb))
    write_new(os.path.join(run_dir, 'eval_job_v2main_001.lsf'), job)
    write_new(os.path.join(run_dir, 'eval_job_diff.txt'), jd)
    return 0


def cmd_pins(args):
    run_dir = os.path.abspath(args.run_dir)
    ns = load_rr5(args.rr5, args.rr5_sha)
    add_fp32_display(ns)
    base = ns.contract_spec(REPO, COMET, OUT)
    eval_dir = os.path.join(run_dir, 'eval_attempt_001')
    observed = json.load(open(args.observed_modules)) if args.observed_modules else None
    closure = import_closure(observed=observed)
    tmp_spec = build_full_spec(ns, base, fp32_matrix_paths(run_dir, ns.METHODS), eval_dir,
                               os.path.join(run_dir, 'pins_v2main_001.json'), 'pending')
    pins = build_pins(open(base['paths']['pins_file'], 'rb').read(), tmp_spec['cells'], eval_dir, closure)
    write_new(os.path.join(run_dir, 'pins_v2main_001.json'), pins)
    print(sha256_bytes(pins))
    return 0


def prepare_full_spec(ns, base, run_dir, pins_path):
    pins_b = open(pins_path, 'rb').read()
    eval_dir = os.path.join(run_dir, 'eval_attempt_001')
    for name, want in (('patient_draws.npy', DRAWS_SHA), ('populations.csv', POPULATIONS_SHA)):
        if base.get('n_reps') == 5000 and sha256_file(os.path.join(eval_dir, name)) != want:
            raise AdapterError(f'eval_attempt_001/{name} sha != pinned')
    spec = build_full_spec(ns, base, fp32_matrix_paths(run_dir, ns.METHODS), eval_dir, pins_path,
                           sha256_bytes(pins_b))
    pre = pre_run_asserts(spec, pins_b)
    return spec, pre


def do_run_a(ns, base, run_dir, run001, n_reps, pins_path):
    out = os.path.join(run_dir, 'rr5_additive_001')
    assert_writable_dir(out)
    spec, pre = prepare_full_spec(ns, base, run_dir, pins_path)
    rc, st = run_rr5(ns, spec, out, os.path.join(run_dir, 'rr5_additive_001.full_spec.json'), n_reps,
                     {'pre_run_asserts': pre})
    require_complete(rc, st, 'Run A')
    orig = [c['cell_id'] for c in spec['cells'] if c['enc'] != FP32]
    fp = [c['cell_id'] for c in spec['cells'] if c['enc'] == FP32]
    rep = bit_identity_report(out, run001, orig, fp)
    rep['rr5_complete'] = {k: st.get(k) for k in ('status', 'n_cells', 'spec_is_contract', 'gates_passed',
                                                 'oracle_winner_count_reproduces_submitted')}
    hg = rep.get('hit_gate_fp32') or {}
    n_hg = len(ns.METHODS) * len(ns.K_VALUES) * spec['n_queries']          # 44,448 on the specification population
    rep['hit_gate_fp32_rows_expected'] = n_hg
    if hg.get('rows_checked') != n_hg or hg.get('hit_mismatches') or hg.get('precision_mismatches'):
        rep['failures'].append(f'hit_gate[virchow2_fp32] {hg} != {n_hg} rows checked with 0 mismatches')
        rep['pass'] = False
    write_new(os.path.join(out, 'bit_identity_report.json'), canonical_json(rep))
    if not rep['pass']:
        raise AdapterError(f'Run A bit identity failed: {rep["failures"][:5]}')
    stats, ctx = recompute_cell_stats(ns, spec, fp, os.path.join(run_dir, 'rr5_additive_001_recompute_bindings'),
                                      n_reps)
    xc = evaluator_crosscheck(ns, stats, os.path.join(run_dir, 'eval_attempt_001', 'estimates.csv'), out,
                              ns.METHODS, ns.K_VALUES)
    xc['hit_gate_fp32'] = rep.get('hit_gate_fp32')
    xc['draws_sha256'] = ctx['draws_sha256']
    write_new(os.path.join(out, 'evaluator_crosscheck.json'), canonical_json(xc))
    if not xc['pass']:
        raise AdapterError(f'evaluator cross-check failed: {xc["failures"][:5]}')
    return spec, rep, xc


def cmd_run_a(args):
    run_dir = os.path.abspath(args.run_dir)
    ns = load_rr5(args.rr5, args.rr5_sha)
    add_fp32_display(ns)
    base = ns.contract_spec(REPO, COMET, OUT)
    do_run_a(ns, base, run_dir, args.run001, args.n_reps, os.path.join(run_dir, 'pins_v2main_001.json'))
    return 0


def check_signoff(signoff, feasibility):
    """A2 sign-off precondition; returns the binding {path, sha256} recorded by Run B (F10)."""
    if not signoff or not os.path.exists(signoff):
        raise AdapterError('Run B requires the lead-signed RUN_B_FEASIBILITY sign-off file (A2)')
    txt = open(signoff).read()
    if sha256_file(feasibility) not in txt or 'option (ii)' not in txt:
        raise AdapterError('sign-off does not name the RUN_B_FEASIBILITY.md sha256 and option (ii)')
    return {'path': os.path.abspath(signoff), 'sha256': sha256_file(signoff),
            'feasibility_path': os.path.abspath(feasibility), 'feasibility_sha256': sha256_file(feasibility)}


_SUMS_LINE = re.compile(r'^([0-9a-f]{64})  ([^/\\\n]+)$')
RUN_A_SUMS_REQUIRED = FINAL_PRODUCTS + ('COMPLETE.json',)


def verify_sha256sums(d, required=RUN_A_SUMS_REQUIRED):
    """F9: `sha256sum -c` equivalent for every line of d/SHA256SUMS.txt (plain names only); every `required`
    product must be listed. Returns {'sha256sums_sha256', 'lines_verified'}; raises AdapterError on any mismatch."""
    p = os.path.join(d, 'SHA256SUMS.txt')
    if not os.path.isfile(p):
        raise AdapterError(f'F9: {p} absent')
    data = open(p, 'rb').read()
    lines = data.decode().split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    if not lines:
        raise AdapterError(f'F9: {p} is empty')
    seen, bad = set(), []
    for ln in lines:
        m = _SUMS_LINE.match(ln)
        if not m:
            raise AdapterError(f'F9: malformed SHA256SUMS line {ln!r}')
        want, name = m.groups()
        if name in seen:
            raise AdapterError(f'F9: duplicate SHA256SUMS entry {name}')
        seen.add(name)
        f = os.path.join(d, name)
        if not os.path.isfile(f):
            bad.append(f'{name}: missing')
        elif sha256_file(f) != want:
            bad.append(f'{name}: sha256 mismatch')
    missing = [n for n in required if n not in seen]
    if missing:
        bad.append(f'not listed: {missing}')
    if bad:
        raise AdapterError(f'F9: Run A SHA256SUMS verification failed: {bad[:5]}')
    return {'sha256sums_sha256': sha256_bytes(data), 'lines_verified': len(lines)}


def do_run_b(rr5_path, rr5_sha, base, run_dir, run001, n_reps, pins_path, signoff_binding=None):
    if not signoff_binding or set(signoff_binding) < {'path', 'sha256'}:
        raise AdapterError('F10: Run B requires the sign-off binding {path, sha256} from check_signoff')
    if sha256_file(signoff_binding['path']) != signoff_binding['sha256']:
        raise AdapterError('F10: sign-off file changed since check_signoff')
    run_a = os.path.join(run_dir, 'rr5_additive_001')
    st_a = json.load(open(os.path.join(run_a, 'COMPLETE.json')))
    if st_a.get('status') != 'COMPLETE' or os.path.exists(os.path.join(run_a, 'FAILED.json')):
        raise AdapterError('B4: Run A is not COMPLETE')
    for f in ('bit_identity_report.json', 'evaluator_crosscheck.json'):
        if not json.load(open(os.path.join(run_a, f))).get('pass'):
            raise AdapterError(f'B4: Run A {f} did not pass')
    sums_a = verify_sha256sums(run_a)
    run_a_sums = sums_a['sha256sums_sha256']
    rb_root = os.path.join(run_dir, 'rr5_replacement_001')
    assert_writable_dir(rb_root)
    os.makedirs(rb_root)
    # fresh namespace for Run B
    ns = load_rr5(rr5_path, rr5_sha)
    add_fp32_display(ns)
    full_spec, _pre = prepare_full_spec(ns, base, run_dir, pins_path)
    a_spec = json.load(open(os.path.join(run_dir, 'rr5_additive_001.full_spec.json')))['full_spec']
    if canonical_json(a_spec) != canonical_json(full_spec):
        raise AdapterError('B4: Run B full_spec != Run A full_spec')
    rb, rb_diff = make_rb_spec(full_spec)
    pins_sha = sha256_file(pins_path)
    if pins_sha != a_spec['pins']['pins_file']:
        raise AdapterError('B4: pins sha != Run A pins sha')
    rec = {}
    install_alias_wrapper(ns, rec)
    raw = os.path.join(rb_root, 'raw')
    rc, st = run_rr5(ns, rb, raw, os.path.join(rb_root, 'rb_spec.json'), n_reps,
                     {'rb_spec_vs_full_spec': rb_diff, 'run_a_sha256sums_sha256': run_a_sums,
                      'pins_file_sha256': pins_sha, 'option': '(ii)', 'wrapper': WRAPPER_NAME,
                      'run_a_sha256sums_lines_verified': sums_a['lines_verified'],
                      'signoff': {'path': signoff_binding['path'], 'sha256': signoff_binding['sha256']}})
    gates_b = require_complete(rc, st, 'Run B (B3)')
    if rec.get('alias_count') != 16 or rec.get('calls') != 1:
        raise AdapterError(f'Run B alias count {rec.get("alias_count")} / calls {rec.get("calls")}, expected 16 / 1')
    b1, bad = b1_check(raw, run_a)
    # B2: fresh namespace, same wrapper, 38-cell specification spec
    ns2 = load_rr5(rr5_path, rr5_sha)
    add_fp32_display(ns2)
    rec2 = {}
    install_alias_wrapper(ns2, rec2)
    b2_dir = os.path.join(rb_root, 'b2_faithfulness')
    rc2, st2 = run_rr5(ns2, base, b2_dir, os.path.join(rb_root, 'b2_spec.json'), n_reps)
    require_complete(rc2, st2, 'B2')
    if rec2.get('alias_count') != 0:
        raise AdapterError('B2 alias count != 0')
    b2, bad2 = b2_check(b2_dir, run001)
    bad += bad2
    kept, counts, bad3 = a10_drop(raw, run_a)
    bad += bad3
    report = {'option': '(ii)', 'wrapper': WRAPPER_NAME, 'alias_count': rec['alias_count'], 'aliases': rec['aliases'],
              'alias_identity': 'asserted with `is`', 'rb_spec_vs_full_spec': rb_diff,
              'b4': {'run_a_sha256sums_sha256': run_a_sums, 'pins_file_sha256': pins_sha,
                     'run_a_sha256sums_lines_verified': sums_a['lines_verified']},
              'signoff': {'path': signoff_binding['path'], 'sha256': signoff_binding['sha256']},
              'signoff_feasibility': {'path': signoff_binding.get('feasibility_path'),
                                      'sha256': signoff_binding.get('feasibility_sha256')},
              'b1_whole_file_identity': b1, 'b2_faithfulness': b2, 'b2_alias_count': rec2['alias_count'],
              'b3_gates_passed': gates_b, 'a10_row_drop': counts,
              'oracle_winner_count_reproduces_submitted': {'value': st.get('oracle_winner_count_reproduces_submitted'),
                                                           'note': A8_NOTE},
              'rr5_spec_is_contract': st.get('spec_is_contract'), 'failures': bad}
    if bad:
        report['pass'] = False
        write_new(os.path.join(rb_root, 'faithfulness_report.json'), canonical_json(report))
        raise AdapterError(f'Run B gates failed: {bad[:5]}')
    kept, edits = a8_and_major4_edits(kept, run_a, ns.METHODS)
    report['post_call_edits'] = edits
    for name in CSV_PRODUCTS:
        head, lines = kept[name]
        data = join_lines(head, lines)
        write_new(os.path.join(rb_root, name), gz_compress_like_rr5(data) if name.endswith('.gz') else data)
    write_new(os.path.join(rb_root, 'main_config_values.json'), open(os.path.join(raw, 'main_config_values.json'),
                                                                     'rb').read())
    u4 = u4_assert(rb_root)
    report['u4_violations'] = u4
    report['pass'] = not u4
    write_new(os.path.join(rb_root, 'faithfulness_report.json'), canonical_json(report))
    if u4:
        raise AdapterError(f'U-4 assertion failed: {u4[:5]}')
    return report


def cmd_run_b(args):
    binding = check_signoff(args.signoff, args.feasibility)
    ns = load_rr5(args.rr5, args.rr5_sha)
    base = ns.contract_spec(REPO, COMET, OUT)
    run_dir = os.path.abspath(args.run_dir)
    do_run_b(args.rr5, args.rr5_sha, base, run_dir, args.run001, args.n_reps,
             os.path.join(run_dir, 'pins_v2main_001.json'), binding)
    return 0


def require_pinned_rr5(path, sha):
    """F8: the CLI never runs an RR-5 other than the pinned file (overrides exist only as function arguments)."""
    if os.path.abspath(path) != os.path.abspath(RR5_PATH) or sha != RR5_SHA:
        raise AdapterError(f'F8: --rr5/--rr5-sha must equal the pinned {RR5_PATH} / {RR5_SHA}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rr5', default=RR5_PATH)
    ap.add_argument('--rr5-sha', default=RR5_SHA)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('internals')
    p = sub.add_parser('eval-spec')
    p.add_argument('--run-dir', default=os.path.join(ROOT, RUN_STAGE))
    p.add_argument('--queue', required=True)
    p = sub.add_parser('pins')
    p.add_argument('--run-dir', default=os.path.join(ROOT, RUN_STAGE))
    p.add_argument('--observed-modules', default=None)
    for name in ('run-a', 'run-b'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', default=os.path.join(ROOT, RUN_STAGE))
        p.add_argument('--run001', default=RUN001)
        p.add_argument('--n-reps', type=int, default=5000)
        if name == 'run-b':
            p.add_argument('--signoff', required=True)
            p.add_argument('--feasibility', default=os.path.join(ROOT, 'implementation_001', 'RUN_B_FEASIBILITY.md'))
    args = ap.parse_args(argv)
    fn = {'internals': cmd_internals, 'eval-spec': cmd_eval_spec, 'pins': cmd_pins, 'run-a': cmd_run_a,
          'run-b': cmd_run_b}[args.cmd]
    try:
        require_pinned_rr5(args.rr5, args.rr5_sha)
        return fn(args)
    except AdapterError as e:
        print(f'STOP: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
