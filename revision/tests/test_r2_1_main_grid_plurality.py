"""Synthetic-fixture tests for scripts/r2_1_main_grid_plurality.py (RR-5, [decision record] candidate 002).

No real data are read. A small synthetic world (4 groups incl. NORMAL, MISC metadata rows, cross-label
patients, int slide ids, shuffled matrix axes, 1e6 same-patient sentinels, coarse-rounded Yottixel distances
with mixed-label tie blocks, a TITAN-published matrix with zero-distance ties) is written under pytest's
tmp_path with every specification file name, then bound by a test-built pins file and a `full_spec`. The frozen
vote module (code, not data) is the real hash-pinned file. Run with an explicit --basetemp outside /tmp and
without the cache provider.
"""
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, 'scripts', 'r2_1_main_grid_plurality.py')
VOTE_MODULE = os.path.join(REPO, 'scripts', 'r2_3_internal_failure_tables.py')


def _load_script():
    spec = importlib.util.spec_from_file_location('r2_1_main_grid_plurality_under_test', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_script()
VOTES = M.load_vote_functions(VOTE_MODULE)
PLUR, GEN = VOTES[0], VOTES[1]

GROUPS = ['ALPHA', 'BETA', 'GAMMA', 'NORMAL']
ENCS = ('uni2', 'virchow2', 'h_optimus_1')
R_REPS = 200
KS = (1, 3, 5, 10)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fsha(p):
    with open(p, 'rb') as f:
        return sha(f.read())


def wbytes(p, b):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        f.write(b)


# ----------------------------------------------------------------------------------------------
# Independent reference ranking / census (per-query loops; not the script's vectorised code)
# ----------------------------------------------------------------------------------------------

def ref_rank(D, pat, q, reverse=False, tiekey=None):
    """Eligible library of query q sorted by (distance, +/-tiekey); tiekey defaults to reference_index."""
    n = D.shape[0]
    r = np.array([j for j in range(n) if pat[j] != pat[q]])
    key = np.arange(n) if tiekey is None else np.asarray(tiekey)
    sec = -key[r] if reverse else key[r]
    return r[np.lexsort((sec, D[q, r]))]


def ref_census(D, pat, y):
    n = D.shape[0]
    out = {'rank1_tie': 0, 'zero_top10': 0, 'zero_entries': 0}
    for k in KS:
        out[f'boundary_tie_{k}'] = 0
        out[f'order_sensitive_{k}'] = 0
    sens = {k: np.zeros(n, bool) for k in KS}
    for q in range(n):
        o = ref_rank(D, pat, q)
        d = D[q, o]
        out['rank1_tie'] += int(d[0] == d[1])
        for k in KS:
            out[f'boundary_tie_{k}'] += int(d[k - 1] == d[k])
            # maximal equal blocks intersecting ranks 1..k with >= 2 labels
            s = False
            i = 0
            while i < k:
                j = i
                while j + 1 < len(d) and d[j + 1] == d[i]:
                    j += 1
                if len(set(y[o[i:j + 1]].tolist())) >= 2:
                    s = True
                i = j + 1
            sens[k][q] = s
            out[f'order_sensitive_{k}'] += int(s)
        out['zero_top10'] += int((d[:10] == 0.0).any())
        out['zero_entries'] += int((d == 0.0).sum())
    return out, sens


def ref_cm(y, pred, G):
    cm = np.zeros((G, G), np.int64)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    return cm


def ref_metrics(cm):
    G = cm.shape[0]
    f1, pr, rc, pc = [], [], [], {}
    for g in range(G):
        tp, p, s = cm[g, g], cm[:, g].sum(), cm[g, :].sum()
        pg = tp / p if p else 0.0
        rg = tp / s if s else 0.0
        fg = 2 * tp / (p + s) if p + s else 0.0
        f1.append(fg)
        pr.append(pg)
        rc.append(rg)
        pc[GROUPS[g]] = {'f1': float(fg), 'precision': float(pg), 'recall': float(rg), 'support': int(s)}
    return {'macro_f1': float(np.mean(f1)), 'micro_f1': float(np.trace(cm) / cm.sum()),
            'macro_precision': float(np.mean(pr)), 'macro_recall': float(np.mean(rc)), 'per_class': pc}


def ref_predict(D, pat, y, reverse=False, tiekey=None):
    """Per k: plurality, oracle (direct calls to the hash-loaded vote module), hit, c_k, top-1."""
    n = D.shape[0]
    res = {k: {'plurality': [], 'oracle': [], 'hit': [], 'c': [], 'top1': []} for k in KS}
    for q in range(n):
        o = ref_rank(D, pat, q, reverse, tiekey)
        labs = [GROUPS[y[j]] for j in o[:10]]
        for k in KS:
            res[k]['plurality'].append(GROUPS.index(PLUR(labs, k)))
            res[k]['oracle'].append(GROUPS.index(GEN(labs, GROUPS[y[q]], k)[1]))
            c = sum(1 for j in o[:k] if y[j] == y[q])
            res[k]['c'].append(c)
            res[k]['hit'].append(int(c > 0))
            res[k]['top1'].append(int(y[o[0]]))
    return {k: {a: np.array(b) for a, b in v.items()} for k, v in res.items()}


# ----------------------------------------------------------------------------------------------
# Synthetic world
# ----------------------------------------------------------------------------------------------

def make_population(rng):
    rows = []
    pid = 0
    for g in GROUPS:
        for i in range(8):
            pid += 1
            for _ in range(2 if i < 6 else 1):
                rows.append((g, f'P{pid:03d}'))
    rows += [('ALPHA', 'PX1'), ('BETA', 'PX1'), ('GAMMA', 'PX2'), ('NORMAL', 'PX2')]   # cross-label patients
    misc = [('MISC', f'PM{i}') for i in range(4)] + [('MISC', 'P001')]
    allrows = rows + misc
    ids = rng.choice(np.arange(100000, 999999), size=len(allrows), replace=False)
    perm = rng.permutation(len(allrows))
    md = pd.DataFrame({'Tumor.group': [allrows[i][0] for i in perm],
                       'Slide.Scan.File': [str(int(ids[i])) for i in perm],
                       'Sample.ID': [f'S{i}' for i in range(len(perm))],
                       'UID.Subject': [allrows[i][1] for i in perm]})
    return md


def make_world(root, seed=7, dup=None):
    """dup: {cell_id: source_cell_id} gives identical matrices (exact float64 winner ties)."""
    rng = np.random.default_rng(seed)
    repo = os.path.join(root, 'repo')
    cfg = os.path.join(root, 'comet-feature-gen')
    out = os.path.join(root, 'prosper_revision_output')
    spec = M.contract_spec(repo, cfg, out)
    spec['name'] = 'synthetic test world'
    spec['paths']['vote_module'] = VOTE_MODULE
    keep_enc = {'uni2', 'virchow2', 'h_optimus_1', 'titan'}
    spec['cells'] = [c for c in spec['cells'] if c['enc'] in keep_enc]
    spec['contested_encoders'] = ['uni2', 'virchow2']
    md = make_population(rng)
    keep = md['Tumor.group'] != 'MISC'
    mdk = md[keep]
    n = int(keep.sum())
    ids = mdk['Slide.Scan.File'].tolist()
    pat = mdk['UID.Subject'].to_numpy()
    lab = mdk['Tumor.group'].to_numpy()
    y = np.array([GROUPS.index(g) for g in lab])
    pop = pd.DataFrame({'query_index': np.arange(n), 'reference_index': np.arange(n),
                        'metadata_index': np.flatnonzero(keep.to_numpy()), 'slide_id': [int(x) for x in ids],
                        'patient_id': pat, 'diagnostic_group': lab})
    all_ids = md['Slide.Scan.File'].tolist()
    all_pat = md['UID.Subject'].to_numpy()
    N = len(all_ids)
    cent = rng.normal(0, 1.0, size=(len(GROUPS) + 1, 6))
    gi = [GROUPS.index(g) if g in GROUPS else len(GROUPS) for g in md['Tumor.group']]
    base = cent[gi] + rng.normal(0, 0.9, size=(N, 6))
    # TITAN published: three different-patient slides with identical embeddings (2 groups) -> zero ties
    pos = {i: j for j, i in enumerate(all_ids)}
    trio = [ids[np.flatnonzero((y == 0) & (pat == 'P001'))[0]], ids[np.flatnonzero((y == 0) & (pat == 'P002'))[0]],
            ids[np.flatnonzero((y == 1) & (pat == 'P010'))[0]]]
    D_pop, mats = {}, {}
    for c in spec['cells']:
        cid = c['cell_id']
        f = base + rng.normal(0, 0.35, size=base.shape)
        if dup and cid in dup:
            mats[cid] = mats[dup[cid]].copy()
            D_pop[cid] = D_pop[dup[cid]].copy()
            continue
        if cid == 'titan/titan_published':
            for t in trio[1:]:
                f[pos[t]] = f[pos[trio[0]]]
        A = np.sqrt(((f[:, None, :] - f[None, :, :]) ** 2).sum(-1))
        if c['method'] == 'yottixel':
            A = np.round(A * 1.5) / 1.5
        A = (A + A.T) / 2.0
        same = all_pat[:, None] == all_pat[None, :]
        A[same] = M.SENTINEL
        perm = rng.permutation(N)                              # matrix axis order != population order
        df = pd.DataFrame(A[np.ix_(perm, perm)], index=[all_ids[i] for i in perm],
                          columns=[all_ids[i] for i in perm])
        mats[cid] = df
        Dp = df.loc[ids, ids].to_numpy().copy()
        Dp[pat[:, None] == pat[None, :]] = np.inf
        D_pop[cid] = Dp
    w = {'root': root, 'spec': spec, 'md': md, 'pop': pop, 'ids': ids, 'pat': pat, 'lab': lab, 'y': y, 'n': n,
         'D': D_pop, 'mats': mats}
    w['census'], w['sens'] = {}, {}
    for cid, Dp in D_pop.items():
        w['census'][cid], w['sens'][cid] = ref_census(Dp, pat, y)
    w['pred'] = {cid: {o: ref_predict(Dp, pat, y, reverse=(o == 'reversed')) for o in ('canonical', 'reversed')}
                 for cid, Dp in D_pop.items()}
    P = sorted(set(pat.tolist()))
    pc = {p: j for j, p in enumerate(P)}
    pcol = np.array([pc[p] for p in pat])
    for s in range(seed, seed + 200):
        dr = np.random.default_rng(s).multinomial(len(P), [1.0 / len(P)] * len(P), size=R_REPS).astype(np.int64)
        sup = dr[:, pcol] @ M.onehot(y, len(GROUPS))
        if (sup > 0).all():
            break
    w['draws'], w['P'], w['pcol'] = dr, P, pcol
    write_world(w)
    return w


def hist_payload(w, cid, k):
    """Historical JSON: canonical ORACLE CM for order-invariant rows; reversed-order CM for sensitive rows."""
    sens_row = w['sens'][cid][k].any()
    pr = w['pred'][cid]['reversed' if sens_row else 'canonical'][k]['oracle']
    cm = ref_cm(w['y'], pr, len(GROUPS))
    # 12-decimal values round-trip exactly through openpyxl (16 significant digits) and stay within 1e-9
    rd = lambda v: round(float(v), 12)  # noqa: E731
    pe = ref_metrics(cm)
    pe = {m: (rd(v) if m != 'per_class' else {g: {a: (rd(b) if a != 'support' else b) for a, b in d.items()}
                                               for g, d in v.items()}) for m, v in pe.items()}
    ci = {m: [rd(pe[m] - 0.05), rd(pe[m] + 0.04)] for m in ('macro_f1', 'micro_f1', 'macro_precision',
                                                              'macro_recall')}
    ci['per_class_f1'] = {g: [rd(pe['per_class'][g]['f1'] * 0.9), rd(min(1.0, pe['per_class'][g]['f1'] + 0.03))]
                          for g in GROUPS}
    return {'config': {'top_k': k, 'larger_is_better': False, 'bootstrap_B': 5000, 'seed': 123,
                       'dropped_labels': ['MISC'], 'tie_break': 'generous_majority', 'exclude_same_patient': False,
                       'bootstrap_mode': 'pps_cluster_one_slide'},
            'n_slides': w['n'], 'labels_order': GROUPS, 'confusion_matrix': cm.tolist(), 'point_estimates': pe,
            'confidence_intervals': ci}


def write_xlsx(w, path):
    spec = w['spec']
    sheets = {s: [] for s in ('Overall Metrics', 'Per-Class F1', 'Per-Class F1 CI', 'Per-Class Precision',
                              'Per-Class Recall', 'Per-Class Support')}
    for c in spec['cells']:
        if not c['reconcile']:
            continue
        for k in KS:
            h = w['hist'][(c['cell_id'], k)]
            pe, ci = h['point_estimates'], h['confidence_intervals']
            key = {'Embedding': c['xlsx_key'][0], 'Aggregation': c['xlsx_key'][1], 'Top K': k}
            om = dict(key)
            for m, col in M.OVERALL_METRIC_COLS.items():
                om[col], om[f'{col} CI Low'], om[f'{col} CI High'] = pe[m], ci[m][0], ci[m][1]
            sheets['Overall Metrics'].append(om)
            pcl = pe['per_class']
            sheets['Per-Class F1'].append(dict(key, **{g: pcl[g]['f1'] for g in GROUPS}))
            sheets['Per-Class Precision'].append(dict(key, **{g: pcl[g]['precision'] for g in GROUPS}))
            sheets['Per-Class Recall'].append(dict(key, **{g: pcl[g]['recall'] for g in GROUPS}))
            sheets['Per-Class Support'].append(dict(key, **{g: pcl[g]['support'] for g in GROUPS}))
            d = dict(key)
            for g in GROUPS:
                d[f'{g} CI Low'], d[f'{g} CI High'] = ci['per_class_f1'][g]
            sheets['Per-Class F1 CI'].append(d)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for s, rows in sheets.items():
            pd.DataFrame(rows).to_excel(xw, sheet_name=s, index=False)


def qm_frame(w, enc, tiekey=None):
    rows = []
    for m in M.METHODS:
        cid = f'{enc}/{m}'
        pr = w['pred'][cid]['canonical'] if tiekey is None else \
            {k: v for k, v in ref_predict(w['D'][cid], w['pat'], w['y'], tiekey=tiekey[cid]).items()}
        for k in KS:
            rows.append(pd.DataFrame({'method': m, 'query_index': np.arange(w['n']),
                                      'query': [int(x) for x in w['ids']], 'diagnostic_group': w['lab'], 'k': k,
                                      'status': 'returned', 'returned_depth': k, 'coverage': 1,
                                      'conditional_hit': pr[k]['hit'], 'conditional_precision': pr[k]['c'] / k}))
    df = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=3)   # row order is not relied on
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode())
    return buf.getvalue()


def r23_frame(w):
    p = w['pred']['uni2/ot_cosine']['canonical']
    G = np.array(GROUPS, dtype=object)
    df = pd.DataFrame({'query': [int(x) for x in w['ids']], 'true_label': G[w['y']], 'top1': G[p[1]['top1']],
                       'plurality_k3': G[p[3]['plurality']], 'plurality_k5': G[p[5]['plurality']],
                       'plurality_k10': G[p[10]['plurality']], 'legacy_generous_k5_pred': G[p[5]['oracle']],
                       'legacy_generous_k5_correct': (p[5]['oracle'] == w['y']).astype(int)})
    for k in KS:
        df[f'hit{k}'] = p[k]['hit']
    for k in (5, 10):
        df[f'precision{k}'] = p[k]['c'] / k
    return df.iloc[::-1]                                     # join on slide id, not row order


def write_world(w):
    spec = w['spec']
    pth = spec['paths']
    wbytes(pth['metadata'], w['md'].to_csv(index=False).encode())
    popb = w['pop'].to_csv(index=False).encode()
    for n in ('population', 'population_copy_virchow2', 'population_copy_h_optimus_1'):
        wbytes(pth[n], popb)
    b = io.BytesIO()
    np.save(b, w['draws'])
    for n in ('draws', 'draws_copy_virchow2', 'draws_copy_h_optimus_1'):
        wbytes(pth[n], b.getvalue())
    for e in ENCS:
        wbytes(pth[f'summary_{e}'], json.dumps({'policy': {'patient_order': w['P']}, 'encoder': e}).encode())
        wbytes(pth[f'qm_{e}'], qm_frame(w, e))
    for c in spec['cells']:
        wbytes(c['path'], pickle.dumps(w['mats'][c['cell_id']]))
    w['hist'] = {}
    for c in spec['cells']:
        if c['reconcile']:
            for k in KS:
                h = hist_payload(w, c['cell_id'], k)
                w['hist'][(c['cell_id'], k)] = h
                wbytes(os.path.join(pth['comet_root'], M.hist_key(c, k)), json.dumps(h).encode())
    write_xlsx(w, pth['xlsx'])
    wbytes(pth['r23_frame'], r23_frame(w).to_csv(index=False).encode())
    # specification-shaped expectations derived from the synthetic data
    spec['n_queries'], spec['n_patients'], spec['n_groups'] = w['n'], len(w['P']), len(GROUPS)
    spec['n_cross_label'] = 2
    spec['n_reps'] = R_REPS
    spec['n_hist_rows'] = sum(1 for c in spec['cells'] if c['reconcile']) * len(KS)
    cen = w['census']
    anchors = {'titan/titan_published': {k: cen['titan/titan_published'][k] for k in
                                         ('rank1_tie', 'boundary_tie_1', 'boundary_tie_3', 'boundary_tie_5',
                                          'boundary_tie_10', 'zero_top10', 'zero_entries')},
               'titan/titan_repaired': {k: 0 for k in cen['titan/titan_repaired']
                                        if k != 'min_eligible_per_query'},
               'uni2/yottixel': {k: cen['uni2/yottixel'][k] for k in ('rank1_tie', 'boundary_tie_5')}}
    for c in spec['cells']:
        if c['method'] in ('median_pooling', 'ot_cosine', 'sampler') and c['enc'] != 'titan':
            anchors.setdefault(c['cell_id'], {})['order_sensitive_10'] = 0
    spec['anchors'] = anchors
    # fig2c: submitted order = descending mean of the three xlsx pipelines' k=5 ORACLE per-class F1
    f = spec['fig2c']
    by_x = {tuple(c['xlsx_key']): c['cell_id'] for c in spec['cells'] if c['xlsx_key']}
    mean = {g: np.mean([w['hist'][(by_x[tuple(x)], 5)]['point_estimates']['per_class'][g]['f1']
                        for x in f['xlsx_pipelines']]) for g in GROUPS}
    f['submitted_order'] = sorted(GROUPS, key=lambda g: (-mean[g], GROUPS.index(g)))
    rebind(w)


def rebind(w):
    """Recompute every pin from the files on disk and rewrite the test pins file (mutations reach later gates)."""
    spec = w['spec']
    pth = spec['paths']
    pn = spec['pins']
    pn['metadata'] = fsha(pth['metadata'])
    pn['population'] = fsha(pth['population'])
    pn['draws'] = fsha(pth['draws'])
    pn['xlsx'] = fsha(pth['xlsx'])
    pn['vote_module'] = fsha(pth['vote_module'])
    pn['r23_frame'] = fsha(pth['r23_frame'])
    pn['qm'] = {e: fsha(pth[f'qm_{e}']) for e in ENCS}
    pn['summary'] = {e: fsha(pth[f'summary_{e}']) for e in ENCS}
    for c in spec['cells']:
        c['sha256'] = fsha(pth.get(f'matrix:{c["cell_id"]}', c['path']))
    pf = {'metadata': pn['metadata'], 'populations': pn['population'], 'table_s3_xlsx': pn['xlsx'],
          'vote_module': pn['vote_module'], 'r2_3_all_query_frame': pn['r23_frame'],
          'matrices': {c['cell_id']: {'path': c['path'], 'sha256': c['sha256']} for c in spec['cells']},
          'packages': {e: {'draws_sha256': pn['draws'], 'populations_sha256': pn['population'],
                           'query_metrics_sha256': pn['qm'][e], 'summary_sha256': pn['summary'][e]} for e in ENCS},
          'historical_json': {M.hist_key(c, k): fsha(os.path.join(pth['comet_root'], M.hist_key(c, k)))
                              for c in spec['cells'] if c['reconcile'] for k in KS}}
    b = json.dumps(pf, indent=1, sort_keys=True).encode()
    wbytes(pth['pins_file'], b)
    pn['pins_file'] = sha(b)
    inv = [f'{c["cell_id"]}@{k}' for c in spec['cells'] if c['reconcile'] and c['method'] != 'yottixel'
           and c['enc'] != 'titan' for k in KS]
    sen = [f'{c["cell_id"]}@{k}' for c in spec['cells'] if c['reconcile'] and (c['method'] == 'yottixel'
           or c['enc'] == 'titan') for k in KS]
    spec['expected_partition'] = {'invariant': inv, 'sensitive': sen}


def run_cli(w, name, extra=(), spec=None):
    out = os.path.join(w['root'], name)
    sp = os.path.join(w['root'], f'{name}_spec.json')
    with open(sp, 'w') as f:
        json.dump({'full_spec': spec or w['spec']}, f)
    rc = M.main(['--out-dir', out, '--spec', sp, '--n-reps', str(R_REPS), *extra])
    return rc, out


def assert_failed(w, name, gate, extra=(), spec=None):
    rc, out = run_cli(w, name, extra, spec)
    assert rc == 1
    assert not os.path.exists(os.path.join(out, 'COMPLETE.json'))
    js = json.load(open(os.path.join(out, 'FAILED.json')))
    assert js['gate'] == gate, js
    assert js['complete_written'] is False and js['automatic_retry'] is False
    assert os.path.exists(os.path.join(out, 'SHA256SUMS.txt'))
    return js, out


@pytest.fixture
def world(tmp_path):
    return make_world(str(tmp_path))


# ----------------------------------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------------------------------

def test_complete_run_outputs_predictions_and_bootstrap(world):
    w = world
    rc, out = run_cli(w, 'run_ok')
    assert rc == 0
    assert os.path.exists(os.path.join(out, 'COMPLETE.json'))
    assert not os.path.exists(os.path.join(out, 'FAILED.json'))
    comp = json.load(open(os.path.join(out, 'COMPLETE.json')))
    assert comp['status'] == 'COMPLETE' and comp['bootstrap'] is True and comp['n_reps'] == R_REPS
    assert comp['n_cell_k_rule_order_evaluated'] == len(w['spec']['cells']) * 4 * 2 * 2
    assert comp['spec_is_contract'] is False
    assert comp['gates_passed'] == ['scope', 'binding', 'draws_binding', 'matrix_structure', 'tie_census',
                                    'tie_logic', 'rule_identity', 'reconciliation_partition', 'reconciliation',
                                    'hit_gate', 'main_config_crosscheck', 'fig2c_identification', 'bootstrap']
    for f in ('input_bindings.json', 'table_s3_revised.csv', 'per_class.csv', 'hit_precision.csv',
              'rule_contrasts.csv', 'aggregation_contrasts.csv', 'titan_contrast.csv', 'winners.csv',
              'fig2a_panel.csv', 'fig2b_matrix.csv', 'fig2c_groups.csv', 'tie_census.csv', 'tie_sensitivity.csv',
              'per_query_predictions.csv.gz', 'reconciliation.json', 'lineage.json', 'main_config_values.json',
              'run_resources.json', 'SHA256SUMS.txt'):
        assert os.path.exists(os.path.join(out, f)), f
    # every input bound by full sha256, matching the files on disk
    ib = json.load(open(os.path.join(out, 'input_bindings.json')))
    flat = json.dumps(ib)
    for c in w['spec']['cells']:
        assert c['sha256'] in flat
    assert fsha(VOTE_MODULE) in flat and w['spec']['pins']['xlsx'] in flat
    # SHA256SUMS verifies every listed output
    for line in open(os.path.join(out, 'SHA256SUMS.txt')).read().splitlines():
        h, name = line.split('  ', 1)
        assert fsha(os.path.join(out, name)) == h, name
    # per-query predictions equal the independent per-query ranking + direct vote-module calls
    pq = pd.read_csv(os.path.join(out, 'per_query_predictions.csv.gz'), dtype={'slide_id': str})
    G = np.array(GROUPS, dtype=object)
    for cid, pr in w['pred'].items():
        for k in KS:
            sub = pq[(pq.cell == cid) & (pq.k == k)].sort_values('query_index')
            assert sub.slide_id.tolist() == w['ids']
            assert (sub.pred_plurality.to_numpy() == G[pr['canonical'][k]['plurality']]).all(), (cid, k)
            assert (sub.pred_oracle.to_numpy() == G[pr['canonical'][k]['oracle']]).all(), (cid, k)
            assert (sub.pred_plurality_reversed.to_numpy() == G[pr['reversed'][k]['plurality']]).all(), (cid, k)
            assert (sub.pred_oracle_reversed.to_numpy() == G[pr['reversed'][k]['oracle']]).all(), (cid, k)
            assert (sub.hit.to_numpy() == pr['canonical'][k]['hit']).all()
            assert (sub.c_k.to_numpy() == pr['canonical'][k]['c']).all()
            assert (sub.order_sensitive_k.to_numpy().astype(bool) == w['sens'][cid][k]).all()
    # tie census equals the independent census; matrix axes were shuffled
    tc = pd.read_csv(os.path.join(out, 'tie_census.csv')).set_index('cell_id')
    for cid, cen in w['census'].items():
        for key, v in cen.items():
            assert tc.loc[cid, key] == v, (cid, key)
        assert not tc.loc[cid, 'matrix_axis_order_equals_population']
    # Table S3 revised: placement of submitted values
    t3 = pd.read_csv(os.path.join(out, 'table_s3_revised.csv'))
    assert len(t3) == len(w['spec']['cells']) * 4 * 2 * 4 * 2
    pub = t3[(t3.cell_id == 'titan/titan_published') & (t3.rule == 'oracle')]
    rep = t3[(t3.cell_id == 'titan/titan_repaired') & (t3.rule == 'oracle')]
    hop = t3[t3.cell_id.str.startswith('h_optimus_1/')]
    assert pub.submitted_value.notna().all() and pub.submitted_ci_low.notna().all()
    assert rep.submitted_value.isna().all() and rep.submitted_ci_low.isna().all()
    assert rep.submitted_value_from_published_matrix.notna().all()
    assert hop[['submitted_value', 'submitted_value_from_published_matrix']].isna().all().all()
    assert t3[t3.rule == 'plurality'].submitted_value.isna().all()
    for _, r in t3[(t3.cell_id == 'uni2/ot_cosine') & (t3.rule == 'oracle')].iterrows():
        assert r.submitted_value == round(w['hist'][('uni2/ot_cosine', int(r.k))]['point_estimates'][r.metric], 6)
    for _, r in rep.iterrows():
        assert r.submitted_value_from_published_matrix == \
            round(w['hist'][('titan/titan_published', int(r.k))]['point_estimates'][r.metric], 6)
    # lineage: TITAN rows reference the published matrix only
    lin = json.load(open(os.path.join(out, 'lineage.json')))
    pub_sha = [c for c in w['spec']['cells'] if c['cell_id'] == 'titan/titan_published'][0]['sha256']
    trows = [r for r in lin['rows'] if r['cell_id'].startswith('titan/')]
    assert trows and all(r['cell_id'] == 'titan/titan_published' and r['matrix_sha256'] == pub_sha for r in trows)
    assert not any(r['cell_id'].startswith('h_optimus_1/') for r in lin['rows'])
    rec = json.load(open(os.path.join(out, 'reconciliation.json')))
    assert rec['titan_repaired_reconciled'] is False and rec['h_optimus_1_reconciled'] is False
    assert rec['n_rows'] == rec['n_rows_passed'] == w['spec']['n_hist_rows']
    # k = 1 identity: plurality - ORACLE contrasts are exactly zero
    rcn = pd.read_csv(os.path.join(out, 'rule_contrasts.csv'))
    k1 = rcn[rcn.k == 1]
    assert len(k1) and (k1.value == 0).all() and (k1.ci_low.fillna(0) == 0).all() and (k1.ci_high.fillna(0) == 0).all()


def _loop_ci(w, pred, metric):
    """Per-replicate loop: patient-weighted confusion matrix -> metric; linear-interpolation percentiles."""
    G = len(GROUPS)
    y = w['y']
    vals = []
    for r in range(R_REPS):
        wt = w['draws'][r, w['pcol']].astype(float)
        cm = np.zeros((G, G))
        np.add.at(cm, (y, pred), wt)
        tp, p, s = np.diag(cm), cm.sum(0), cm.sum(1)
        f1 = np.array([2 * tp[g] / (p[g] + s[g]) if p[g] + s[g] else 0.0 for g in range(G)])
        vals.append(f1.mean() if metric == 'macro_f1' else tp.sum() / s.sum())
    return np.percentile(vals, [2.5, 97.5])


def test_bootstrap_matches_replicate_loop(world):
    w = world
    rc, out = run_cli(w, 'run_ci')
    assert rc == 0
    t3 = pd.read_csv(os.path.join(out, 'table_s3_revised.csv'))
    for cid, rule, order, k in (('uni2/ot_cosine', 'plurality', 'canonical', 5),
                                ('virchow2/yottixel', 'oracle', 'reversed', 10),
                                ('titan/titan_published', 'plurality', 'canonical', 3)):
        pred = w['pred'][cid][order][k][rule]
        for metric in ('macro_f1', 'micro_f1'):
            row = t3[(t3.cell_id == cid) & (t3.rule == rule) & (t3.tie_order == order) & (t3.k == k)
                     & (t3.metric == metric)].iloc[0]
            pm = ref_metrics(ref_cm(w['y'], pred, len(GROUPS)))[metric]
            assert abs(row.value - round(pm, 6)) < 1e-9             # specification: values at 6 dp
            if row.ci_status == 'ok':
                lo, hi = _loop_ci(w, pred, metric)
                assert abs(row.ci_low - round(lo, 6)) < 1e-9 and abs(row.ci_high - round(hi, 6)) < 1e-9, \
                    (cid, rule, metric)
    # per-class support is the patient-weighted truth count per replicate
    pc = pd.read_csv(os.path.join(out, 'per_class.csv'))
    sup = w['draws'][:, w['pcol']] @ M.onehot(w['y'], len(GROUPS))
    for g, name in enumerate(GROUPS):
        row = pc[(pc.cell_id == 'uni2/sampler') & (pc.k == 5) & (pc.rule == 'plurality') &
                 (pc.tie_order == 'canonical') & (pc.group == name) & (pc.metric == 'support')].iloc[0]
        assert row.value == (w['y'] == g).sum()
        lo, hi = np.percentile(sup[:, g], [2.5, 97.5])
        assert abs(row.ci_low - round(lo, 6)) < 1e-9 and abs(row.ci_high - round(hi, 6)) < 1e-9
    # IR5-2: rule-independent hit/precision@k (micro and equal-weight macro) and per-class precision/recall/F1,
    # point and interval, both tie orders, every k, against the naive per-replicate loop
    hp = pd.read_csv(os.path.join(out, 'hit_precision.csv'))
    wts = [np.ones(len(w['y']))] + [w['draws'][r, w['pcol']].astype(float) for r in range(R_REPS)]
    for cid in ('uni2/ot_cosine', 'virchow2/yottixel', 'titan/titan_published'):
        for order in ('canonical', 'reversed'):
            for k in KS:
                pr = w['pred'][cid][order][k]
                ref = _naive_hp(w['y'], len(GROUPS), pr['hit'], pr['c'] / k, wts)
                for metric, vals in ref.items():
                    row = hp[(hp.cell_id == cid) & (hp.tie_order == order) & (hp.k == k)
                             & (hp.metric == metric)].iloc[0]
                    _assert_point_ci(row, vals, 'ok', (cid, order, k, metric))
                for rule in ('plurality', 'oracle'):
                    ref = _naive_per_class(w['y'], pr[rule], len(GROUPS), wts)
                    sub = pc[(pc.cell_id == cid) & (pc.k == k) & (pc.rule == rule) & (pc.tie_order == order)]
                    for g, name in enumerate(GROUPS):
                        for metric in ('precision', 'recall', 'f1'):
                            row = sub[(sub.group == name) & (sub.metric == metric)].iloc[0]
                            vals, nzp = ref[metric][g], ref['n_zero_pred'][g]
                            want = 'not_reportable' if metric != 'recall' and nzp > R_REPS // 100 else 'ok'
                            _assert_point_ci(row, vals, want, (cid, order, k, rule, name, metric))


def _naive_hp(y, G, hit, prec, wts):
    """Per weight vector: micro = weighted sum / weighted support; macro = mean over true groups of per-group
    weighted mean (equal weight). Returns {metric: [point, rep_1, ..., rep_R]}."""
    out = {m: [] for m in ('hit_micro', 'hit_macro', 'precision_micro', 'precision_macro')}
    for wt in wts:
        for name, x in (('hit', hit), ('precision', prec)):
            num = [sum(wt[i] * x[i] for i in range(len(y)) if y[i] == g) for g in range(G)]
            den = [sum(wt[i] for i in range(len(y)) if y[i] == g) for g in range(G)]
            out[f'{name}_micro'].append(sum(num) / sum(den))
            out[f'{name}_macro'].append(np.mean([num[g] / den[g] for g in range(G)]))
    return out


def _naive_per_class(y, pred, G, wts):
    """Per weight vector: weighted confusion matrix -> per-class precision/recall/F1 (zero_division = 0)."""
    out = {m: {g: [] for g in range(G)} for m in ('precision', 'recall', 'f1')}
    nzp = np.zeros(G, int)
    for j, wt in enumerate(wts):
        cm = np.zeros((G, G))
        np.add.at(cm, (y, pred), wt)
        for g in range(G):
            tp, p, s = cm[g, g], cm[:, g].sum(), cm[g, :].sum()
            out['precision'][g].append(tp / p if p else 0.0)
            out['recall'][g].append(tp / s if s else 0.0)
            out['f1'][g].append(2 * tp / (p + s) if p + s else 0.0)
            if j > 0 and p == 0:
                nzp[g] += 1
    out['n_zero_pred'] = nzp
    return out


def _assert_point_ci(row, vals, status, tag):
    assert row.ci_status == status, tag
    assert abs(row.value - round(vals[0], 6)) < 1e-9, tag
    if status == 'ok':
        lo, hi = np.percentile(vals[1:], [2.5, 97.5])
        assert abs(row.ci_low - round(lo, 6)) < 1e-9 and abs(row.ci_high - round(hi, 6)) < 1e-9, tag
    else:
        assert np.isnan(row.ci_low) and np.isnan(row.ci_high), tag


def test_cell_statistics_unbalanced_fixture_matches_naive_loop():
    """IR5-2: unequal group supports, so micro != macro for every rule-independent and overall aggregate; points
    and intervals of hit/precision@k (micro, macro), macro/micro F1, macro precision/recall and per-class
    precision/recall/F1 must equal an independent per-replicate loop."""
    rng = np.random.default_rng(11)
    G, R = 3, 400
    y = np.repeat(np.arange(G), [4, 9, 17])
    n = len(y)
    pat = rng.integers(0, 18, size=n)
    P = sorted(set(pat.tolist()))
    pcol = np.array([P.index(p) for p in pat])
    # patient-multinomial replicates, keeping only those in which every true group has support
    dr = rng.multinomial(len(P), [1.0 / len(P)] * len(P), size=4 * R)
    dr = dr[((dr[:, pcol] @ M.onehot(y, G)) > 0).all(axis=1)][:R]
    assert len(dr) == R
    W = dr[:, pcol].astype(float)
    T = M.onehot(y, G)
    preds = {}
    for o in M.ORDERS:
        for k in M.K_VALUES:
            c = rng.integers(0, k + 1, size=n) * rng.integers(0, 2, size=n)
            preds[(o, k)] = {'plurality': rng.integers(0, G, size=n), 'oracle': rng.integers(0, G, size=n),
                             'hit': (c > 0).astype(np.int64), 'c': c.astype(np.int64)}
    st = M.cell_statistics(W, T, W @ T, T.sum(axis=0), y, preds, G)
    wts = [np.ones(n)] + [W[r] for r in range(R)]

    def check(s, vals, tag, ok=True):
        assert abs(s.point - vals[0]) < 1e-12, tag
        if ok:
            lo, hi = np.percentile(vals[1:], [2.5, 97.5])
            assert abs(s.lo - lo) < 1e-12 and abs(s.hi - hi) < 1e-12, tag

    for o in M.ORDERS:
        for k in M.K_VALUES:
            p = preds[(o, k)]
            ref = _naive_hp(y, G, p['hit'], p['c'] / k, wts)
            assert abs(ref['hit_micro'][0] - ref['hit_macro'][0]) > 1e-6 or \
                abs(ref['precision_micro'][0] - ref['precision_macro'][0]) > 1e-6
            for m, vals in ref.items():
                check(st[(o, k, 'hp')][m], vals, (o, k, m))
            for rule in M.RULES:
                d = st[(o, k, rule)]
                pcr = _naive_per_class(y, p[rule], G, wts)
                thr = R // 100
                for g in range(G):
                    for m in ('precision', 'recall', 'f1'):
                        ok = m == 'recall' or pcr['n_zero_pred'][g] <= thr
                        assert d['per_class'][g][m].status == ('ok' if ok else 'not_reportable'), (o, k, rule, g, m)
                        check(d['per_class'][g][m], pcr[m][g], (o, k, rule, g, m), ok)
                macro = {m: [np.mean([pcr[m][g][j] for g in range(G)]) for j in range(R + 1)]
                         for m in ('precision', 'recall', 'f1')}
                micro = []
                for wt in wts:
                    micro.append(sum(wt[i] for i in range(n) if p[rule][i] == y[i]) / wt.sum())
                n_any = sum(1 for j in range(1, R + 1)
                            if any(sum(wts[j][i] for i in range(n) if p[rule][i] == g) == 0 for g in range(G)))
                macro_ok = n_any <= thr
                assert d['macro_f1'].status == d['macro_precision'].status == \
                    ('ok' if macro_ok else 'not_reportable'), (o, k, rule)
                check(d['macro_f1'], macro['f1'], (o, k, rule, 'macro_f1'), macro_ok)
                check(d['macro_precision'], macro['precision'], (o, k, rule, 'macro_precision'), macro_ok)
                check(d['macro_recall'], macro['recall'], (o, k, rule, 'macro_recall'))
                check(d['micro_f1'], micro, (o, k, rule, 'micro_f1'))


def test_gates_only_writes_no_metrics(world):
    w = world
    rc, out = run_cli(w, 'run_go', extra=('--gates-only',))
    assert rc == 0
    names = set(os.listdir(out))
    assert {'GATES_ONLY.json', 'input_bindings.json', 'SHA256SUMS.txt'} <= names
    assert 'COMPLETE.json' not in names and 'FAILED.json' not in names and 'table_s3_revised.csv' not in names
    js = json.load(open(os.path.join(out, 'GATES_ONLY.json')))
    assert js['status'] == 'GATES_ONLY_PASS' and 'bootstrap' not in js['gates_passed']
    assert 'macro_f1' not in json.dumps(js)


def test_nonempty_out_dir_is_refused(world):
    w = world
    out = os.path.join(w['root'], 'busy')
    os.makedirs(out)
    open(os.path.join(out, 'x'), 'w').close()
    sp = os.path.join(w['root'], 'busy_spec.json')
    json.dump({'full_spec': w['spec']}, open(sp, 'w'))
    assert M.main(['--out-dir', out, '--spec', sp, '--n-reps', str(R_REPS)]) == 2
    assert sorted(os.listdir(out)) == ['x']


# ----------------------------------------------------------------------------------------------
# Ranking, tie blocks and census (unit level, constructed rows)
# ----------------------------------------------------------------------------------------------

def _single_query(dist0, labels, qlabel=3):
    """Query 0 (label qlabel) against references 1..n-1 with the given distances; other rows distinct."""
    n = len(dist0) + 1
    rng = np.random.default_rng(0)
    X = rng.permutation(np.arange(1, n * n + 1, dtype=float)).reshape(n, n) + 100.0
    X[0, 1:] = dist0
    same = np.eye(n, dtype=bool)
    X[same] = np.inf
    y = np.array([qlabel] + list(labels))
    return X, same, y


def test_internal_nonboundary_tie_block_changes_plurality_tiebreak():
    # ranks: A(1) [B C](2,2) B(3) C(4) | A(5) A(6) ...: k=5 counts B=2, C=2 -> closest-ranked tied label decides
    X, same, y = _single_query([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 1, 2, 0, 0, 3, 3, 0, 0, 3])
    top, flags, census = M.rank_cell(X, same, y)
    assert top['canonical'][0].tolist()[:5] == [1, 2, 3, 4, 5]
    assert top['reversed'][0].tolist()[:5] == [1, 3, 2, 4, 5]
    assert not flags['boundary_tie_5'][0] and not flags['rank1_tie'][0]
    assert flags['order_sensitive_5'][0] and flags['order_sensitive_3'][0] and not flags['order_sensitive_1'][0]
    pr = M.predict_cell(top, y, GROUPS, PLUR, GEN)
    assert GROUPS[pr[('canonical', 5)]['plurality'][0]] == 'BETA'
    assert GROUPS[pr[('reversed', 5)]['plurality'][0]] == 'GAMMA'
    assert pr[('canonical', 5)]['c'][0] == pr[('reversed', 5)]['c'][0]
    # a boundary-only criterion would have missed this query: no tie at rank 5/6
    assert X[0, top['canonical'][0][4]] != X[0, top['canonical'][0][5]]


def test_boundary_ties_same_label_vs_mixed_label():
    # same-label boundary block at k=3 (ranks 3-4 both BETA): boundary tie, not order sensitive
    X, same, y = _single_query([1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 2, 1, 1, 0, 2, 0, 2, 0, 2, 0, 2])
    top, flags, census = M.rank_cell(X, same, y)
    assert flags['boundary_tie_3'][0] and not flags['order_sensitive_3'][0]
    # mixed-label boundary block at k=3 (BETA, GAMMA)
    X, same, y = _single_query([1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2])
    top, flags, census = M.rank_cell(X, same, y)
    assert flags['boundary_tie_3'][0] and flags['order_sensitive_3'][0] and not flags['order_sensitive_1'][0]
    assert flags['order_sensitive_5'][0] and flags['order_sensitive_10'][0]
    assert census['boundary_tie_3'] >= 1


def test_zero_distance_ties_counted():
    X, same, y = _single_query([0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 0, 1, 1, 0, 2, 0, 2, 0, 2, 0, 2])
    top, flags, census = M.rank_cell(X, same, y)
    assert census['zero_entries'] == 3 and census['zero_top10'] == 1
    assert flags['rank1_tie'][0] and flags['order_sensitive_1'][0]
    pr = M.predict_cell(top, y, GROUPS, PLUR, GEN)
    assert GROUPS[pr[('canonical', 1)]['plurality'][0]] == 'ALPHA'
    assert GROUPS[pr[('reversed', 1)]['plurality'][0]] == 'BETA'


def test_synthetic_world_census_and_cell_flags(world):
    w = world
    # the synthetic world exercises both orders: Yottixel and TITAN-published predictions differ by order
    diff = sum(int((w['pred'][c]['canonical'][k]['plurality'] != w['pred'][c]['reversed'][k]['plurality']).sum())
               for c in w['pred'] for k in KS)
    assert diff > 0
    assert w['census']['titan/titan_published']['zero_entries'] > 0
    assert all(w['census'][c][f'order_sensitive_{k}'] == 0 for c in w['census'] if not
               (c.endswith('yottixel') or c == 'titan/titan_published') for k in KS)


# ----------------------------------------------------------------------------------------------
# Vote module, rule identity and tie logic
# ----------------------------------------------------------------------------------------------

def test_vote_module_hash_pin_and_rule_equivalence():
    with pytest.raises(M.GateError) as e:
        M.load_vote_functions(VOTE_MODULE, '0' * 64)
    assert e.value.gate == 'binding'
    src = open(VOTE_MODULE, 'rb').read()
    with pytest.raises(M.GateError):
        M.load_vote_functions(VOTE_MODULE, sha(src), source=src + b'\n# mutated\n')
    rng = np.random.default_rng(5)
    n = 300
    top = {'canonical': np.array([rng.permutation(np.arange(1, n))[:10] for _ in range(n)]),
           'reversed': np.array([rng.permutation(np.arange(1, n))[:10] for _ in range(n)])}
    y = rng.integers(0, 4, size=n)
    pr = M.predict_cell(top, y, GROUPS, PLUR, GEN)
    for o in ('canonical', 'reversed'):
        for k in KS:
            for i in range(n):
                labs = [GROUPS[j] for j in y[top[o][i]]]
                assert GROUPS[pr[(o, k)]['plurality'][i]] == PLUR(labs, k)
                c, p = GEN(labs, GROUPS[y[i]], k)
                assert GROUPS[pr[(o, k)]['oracle'][i]] == p and bool(pr[(o, k)]['oracle_module_correct'][i]) == c
    for o in ('canonical', 'reversed'):
        p1 = pr[(o, 1)]
        assert (p1['plurality'] == p1['oracle']).all() and (p1['plurality'] == p1['top1']).all()


def _world_preds(w, cid):
    X = w['D'][cid].copy()
    same = np.isinf(X)
    top, flags, _ = M.rank_cell(X, same, w['y'])
    return M.predict_cell(top, w['y'], GROUPS, PLUR, GEN), flags


def test_rule_identity_and_tie_logic_units(world):
    w = world
    pr, flags = _world_preds(w, 'uni2/yottixel')
    M.check_rule_identity('uni2/yottixel', pr, w['y'])
    M.check_tie_logic('uni2/yottixel', pr, flags)
    bad = copy.deepcopy(pr)
    i = 0
    bad[('canonical', 1)]['oracle'][i] = (bad[('canonical', 1)]['oracle'][i] + 1) % 4
    with pytest.raises(M.GateError) as e:
        M.check_rule_identity('x', bad, w['y'])
    assert e.value.gate == 'rule_identity'
    bad = copy.deepcopy(pr)
    q = int(np.flatnonzero(~flags['order_sensitive_5'])[0])
    bad[('reversed', 5)]['plurality'][q] = (bad[('reversed', 5)]['plurality'][q] + 1) % 4
    with pytest.raises(M.GateError) as e:
        M.check_tie_logic('x', bad, flags)
    assert e.value.gate == 'tie_logic'


def test_corrupted_predictions_fail_tie_logic_then_rule_identity(world, monkeypatch):
    w = world
    orig = M.predict_cell

    def corrupt_reversed(top, y, labels_order, p, g):
        r = orig(top, y, labels_order, p, g)
        r[('reversed', 10)]['plurality'] = (r[('reversed', 10)]['plurality'] + 1) % len(labels_order)
        return r
    monkeypatch.setattr(M, 'predict_cell', corrupt_reversed)
    js, out = assert_failed(w, 'run_tl', 'tie_logic')
    assert js['gates_passed_before_failure'][-1] == 'tie_census'

    def corrupt_both(top, y, labels_order, p, g):
        r = orig(top, y, labels_order, p, g)
        for o in ('canonical', 'reversed'):
            r[(o, 1)]['oracle'] = (r[(o, 1)]['oracle'] + 1) % len(labels_order)
        return r
    monkeypatch.setattr(M, 'predict_cell', corrupt_both)
    js, out = assert_failed(w, 'run_ri', 'rule_identity')
    assert js['gates_passed_before_failure'][-1] == 'tie_logic'


# ----------------------------------------------------------------------------------------------
# Bootstrap interval status (zero predictions / zero support)
# ----------------------------------------------------------------------------------------------

def _status_fixture(m_zero_pred, zero_support=False):
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    G = 4
    R = 200
    W = np.ones((R, len(y)))
    W[:m_zero_pred, 0] = 0.0                 # slide 0 is the only slide predicted as class 2 (plurality)
    if zero_support:
        W[R - 1, 6:8] = 0.0                  # class 3 has no support in one replicate
    T = M.onehot(y, G)
    S_rep, S_pt = W @ T, T.sum(axis=0)
    pl = np.array([2, 0, 1, 1, 1, 1, 3, 3])
    preds = {}
    for o in M.ORDERS:
        for k in M.K_VALUES:
            preds[(o, k)] = {'plurality': pl, 'oracle': y.copy(), 'hit': np.ones(len(y), np.int64),
                             'c': np.ones(len(y), np.int64)}
    return M.cell_statistics(W, T, S_rep, S_pt, y, preds, G)


def test_zero_prediction_replicate_status():
    st = _status_fixture(3)                  # 3 > 200 // 100 -> not reportable
    d = st[('canonical', 5, 'plurality')]
    assert d['per_class'][2]['f1'].status == 'not_reportable' and np.isnan(d['per_class'][2]['f1'].lo)
    assert d['per_class'][2]['f1'].n_zero_pred == 3
    assert d['per_class'][2]['precision'].status == 'not_reportable'
    assert d['per_class'][2]['recall'].status == 'ok'
    assert d['macro_f1'].status == 'not_reportable' and d['macro_precision'].status == 'not_reportable'
    assert d['micro_f1'].status == 'ok' and np.isfinite(d['micro_f1'].lo)
    assert d['macro_recall'].status == 'ok'
    assert st[('canonical', 5, 'oracle')]['macro_f1'].status == 'ok'
    st = _status_fixture(2)                  # 2 <= threshold -> reported (zero predictions counted as F1 0)
    d = st[('canonical', 5, 'plurality')]
    assert d['per_class'][2]['f1'].status == 'ok' and np.isfinite(d['per_class'][2]['f1'].lo)
    assert d['macro_f1'].status == 'ok'
    st = _status_fixture(0, zero_support=True)
    d = st[('canonical', 5, 'oracle')]
    assert d['per_class'][3]['f1'].status == 'zero_support' and np.isnan(d['per_class'][3]['f1'].hi)
    assert d['macro_f1'].status == 'zero_support' and np.isnan(d['macro_f1'].lo)
    assert d['micro_f1'].status == 'ok'
    assert d['per_class'][3]['support'].status == 'ok' and np.isfinite(d['per_class'][3]['support'].lo)
    assert st[('canonical', 5, 'hp')]['hit_macro'].status == 'zero_support'
    assert st[('canonical', 5, 'hp')]['hit_micro'].status == 'ok'
    c = M.contrast_stat(d['macro_f1'], st[('canonical', 5, 'plurality')]['micro_f1'])
    assert c.status == 'zero_support'


# ----------------------------------------------------------------------------------------------
# Gate rejections on mutated inputs (gate order respected; FAILED without COMPLETE)
# ----------------------------------------------------------------------------------------------

def test_scope_rejects_forbidden_input_before_binding(world):
    w = world
    spec = copy.deepcopy(w['spec'])
    p = spec['paths']['qm_uni2']
    spec['paths']['qm_uni2'] = os.path.join(os.path.dirname(p), 'tile_embedding_' + os.path.basename(p))
    js, out = assert_failed(w, 'run_scope', 'scope', spec=spec)
    assert js['inputs_bound_so_far'] == {} and js['gates_passed_before_failure'] == []


def test_binding_rejects_mutated_xlsx_and_pins(world):
    w = world
    with open(w['spec']['paths']['xlsx'], 'ab') as f:
        f.write(b'\0')
    js, out = assert_failed(w, 'run_bind_xlsx', 'binding')
    assert js['gates_passed_before_failure'] == ['scope']


def test_binding_rejects_wrong_vote_module_pin(world):
    w = world
    spec = copy.deepcopy(w['spec'])
    spec['pins']['vote_module'] = '1' * 64
    assert_failed(w, 'run_bind_vote', 'binding', spec=spec)


def test_binding_rejects_xlsx_json_disagreement(world):
    w = world
    key = ('uni2/sampler', 3)
    h = copy.deepcopy(w['hist'][key])
    h['point_estimates']['macro_recall'] = round(h['point_estimates']['macro_recall'] + 1e-6, 12)
    c = [c for c in w['spec']['cells'] if c['cell_id'] == key[0]][0]
    wbytes(os.path.join(w['spec']['paths']['comet_root'], M.hist_key(c, key[1])), json.dumps(h).encode())
    rebind(w)
    assert_failed(w, 'run_bind_xj', 'binding')


def test_draws_binding_patient_order_mismatch(world):
    w = world
    po = list(w['P'])
    po[0], po[1] = po[1], po[0]
    wbytes(w['spec']['paths']['summary_virchow2'], json.dumps({'policy': {'patient_order': po}}).encode())
    rebind(w)
    js, out = assert_failed(w, 'run_po', 'draws_binding')
    assert js['gates_passed_before_failure'] == ['scope', 'binding']


def test_draws_binding_copy_and_row_sum(world):
    w = world
    d2 = w['draws'].copy()
    d2[0, 0] += 1
    d2[0, 1] -= 1
    b = io.BytesIO()
    np.save(b, d2)
    wbytes(w['spec']['paths']['draws_copy_h_optimus_1'], b.getvalue())
    rebind(w)
    # every copy is pinned to the original's full sha256, so a differing copy is already rejected at binding
    js, out = assert_failed(w, 'run_dcopy', 'binding')
    assert any('draws_copy_h_optimus_1' in str(d) for d in (js['detail'] or [])) or \
        'draws_copy_h_optimus_1' in js['message']
    d3 = w['draws'].copy()
    d3[5, 0] += 1
    b = io.BytesIO()
    np.save(b, d3)
    for n in ('draws', 'draws_copy_virchow2', 'draws_copy_h_optimus_1'):
        wbytes(w['spec']['paths'][n], b.getvalue())
    rebind(w)
    assert_failed(w, 'run_dsum', 'draws_binding')


def _rewrite_matrix(w, cid, df):
    c = [c for c in w['spec']['cells'] if c['cell_id'] == cid][0]
    wbytes(c['path'], pickle.dumps(df))
    rebind(w)


def test_matrix_structure_rejects_int_labels(world):
    w = world
    df = w['mats']['uni2/sampler'].copy()
    df.index = [int(x) for x in df.index]
    df.columns = [int(x) for x in df.columns]
    _rewrite_matrix(w, 'uni2/sampler', df)
    js, out = assert_failed(w, 'run_intlab', 'matrix_structure')
    assert js['gates_passed_before_failure'][-1] == 'draws_binding'


def test_matrix_structure_rejects_sentinel_and_negative(world):
    w = world
    df = w['mats']['virchow2/ot_cosine'].copy()
    i = w['ids'][np.flatnonzero(w['pat'] == 'P001')[0]]
    j = w['ids'][np.flatnonzero(w['pat'] == 'P001')[1]]
    df.loc[i, j] = 0.5
    _rewrite_matrix(w, 'virchow2/ot_cosine', df)
    assert_failed(w, 'run_sent', 'matrix_structure')
    df = w['mats']['virchow2/ot_cosine'].copy()
    df.loc[w['ids'][0], w['ids'][-1]] = -0.25
    _rewrite_matrix(w, 'virchow2/ot_cosine', df)
    assert_failed(w, 'run_neg', 'matrix_structure')


def test_tie_census_rejects_anchor_mutation(world):
    w = world
    spec = copy.deepcopy(w['spec'])
    spec['anchors']['titan/titan_published']['zero_entries'] += 1
    js, out = assert_failed(w, 'run_anchor', 'tie_census', spec=spec)
    assert js['gates_passed_before_failure'][-1] == 'matrix_structure'


def test_reconciliation_partition_mutation(world):
    w = world
    spec = copy.deepcopy(w['spec'])
    moved = spec['expected_partition']['invariant'].pop(0)
    spec['expected_partition']['sensitive'].append(moved)
    js, out = assert_failed(w, 'run_part', 'reconciliation_partition', spec=spec)
    assert js['gates_passed_before_failure'][-1] == 'rule_identity'


def test_reconciliation_rejects_historical_cm_mutation(world):
    w = world
    key = ('virchow2/median_pooling', 5)
    h = copy.deepcopy(w['hist'][key])
    cm = np.array(h['confusion_matrix'])
    g = int(np.argmax(np.diag(cm)))
    cm[g, g] -= 1
    cm[g, (g + 1) % 4] += 1                  # row sums (support) unchanged
    h['confusion_matrix'] = cm.tolist()
    c = [c for c in w['spec']['cells'] if c['cell_id'] == key[0]][0]
    wbytes(os.path.join(w['spec']['paths']['comet_root'], M.hist_key(c, key[1])), json.dumps(h).encode())
    rebind(w)
    js, out = assert_failed(w, 'run_cm', 'reconciliation')
    assert any('virchow2/median_pooling@5' in d for d in js['detail'])
    assert js['gates_passed_before_failure'][-1] == 'reconciliation_partition'


def test_titan_reconciliation_routes_to_published_matrix(world):
    """Reconciling the submitted TITAN rows against the repaired matrix must fail."""
    w = world
    rep = [c for c in w['spec']['cells'] if c['cell_id'] == 'titan/titan_repaired'][0]
    pub = [c for c in w['spec']['cells'] if c['cell_id'] == 'titan/titan_published'][0]
    # precondition: the repaired ORACLE predictions do not reproduce some submitted (published) CM
    differs = [k for k in KS if not np.array_equal(
        ref_cm(w['y'], w['pred']['titan/titan_repaired']['canonical'][k]['oracle'], 4),
        np.array(w['hist'][('titan/titan_published', k)]['confusion_matrix']))]
    assert differs
    wbytes(pub['path'], open(rep['path'], 'rb').read())       # published path now carries repaired bytes
    spec = w['spec']
    spec['anchors']['titan/titan_published'] = {k: 0 for k in spec['anchors']['titan/titan_published']}
    rebind(w)
    ep = spec['expected_partition']
    for k in KS:
        r = f'titan/titan_published@{k}'
        ep['sensitive'].remove(r)
        ep['invariant'].append(r)
    js, out = assert_failed(w, 'run_titan', 'reconciliation')
    assert all(d.startswith('titan/titan_published@') for d in js['detail'])
    assert len(js['detail']) == len(differs)


def test_hit_gate_rejects_axis_order_tie_breaking(world):
    w = world
    tiekey = {}
    for c in w['spec']['cells']:
        ax = w['mats'][c['cell_id']].index.get_indexer(w['ids'])
        assert (ax >= 0).all()
        tiekey[c['cell_id']] = ax
    # precondition: axis-order tie-breaking changes some hit / c_k against the canonical convention
    changed = 0
    for e in ENCS:
        for m in M.METHODS:
            cid = f'{e}/{m}'
            alt = ref_predict(w['D'][cid], w['pat'], w['y'], tiekey=tiekey[cid])
            changed += sum(int((alt[k]['c'] != w['pred'][cid]['canonical'][k]['c']).sum()) for k in KS)
    assert changed > 0
    for e in ENCS:
        wbytes(w['spec']['paths'][f'qm_{e}'], qm_frame(w, e, tiekey=tiekey))
    rebind(w)
    js, out = assert_failed(w, 'run_axis', 'hit_gate')
    assert js['gates_passed_before_failure'][-1] == 'reconciliation'


def test_hit_gate_rejects_single_flipped_hit(world):
    w = world
    p = w['spec']['paths']['qm_h_optimus_1']
    df = pd.read_csv(p, compression='gzip')
    i = int(np.flatnonzero(((df.k == 3) & (df.method == 'sampler')).to_numpy())[0])
    df.loc[i, 'conditional_hit'] = 1 - df.loc[i, 'conditional_hit']
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode())
    wbytes(p, buf.getvalue())
    rebind(w)
    assert_failed(w, 'run_hitflip', 'hit_gate')


def test_main_config_crosscheck_rejects_frame_mutation(world):
    w = world
    p = w['spec']['paths']['r23_frame']
    df = pd.read_csv(p, dtype={'query': str})
    df.loc[0, 'plurality_k5'] = [g for g in GROUPS if g != df.loc[0, 'plurality_k5']][0]
    wbytes(p, df.to_csv(index=False).encode())
    rebind(w)
    js, out = assert_failed(w, 'run_mcc', 'main_config_crosscheck')
    assert js['gates_passed_before_failure'][-1] == 'hit_gate'


def test_fig2c_identification_rejects_order_mutation(world):
    w = world
    spec = copy.deepcopy(w['spec'])
    so = spec['fig2c']['submitted_order']
    so[0], so[-1] = so[-1], so[0]
    js, out = assert_failed(w, 'run_f2c', 'fig2c_identification', spec=spec)
    assert js['gates_passed_before_failure'][-1] == 'main_config_crosscheck'


def test_bootstrap_rejects_wrong_replicate_count_before_computation(world):
    """IR5-4: a --n-reps differing from the spec is refused right after draws_binding, before any metric table."""
    w = world
    rc, out = run_cli(w, 'run_boot', extra=('--n-reps', str(R_REPS - 1)))
    assert rc == 1
    assert not os.path.exists(os.path.join(out, 'COMPLETE.json'))
    js = json.load(open(os.path.join(out, 'FAILED.json')))
    assert js['gate'] == 'bootstrap'
    assert js['gates_passed_before_failure'][-1] == 'draws_binding'
    assert not any(n.endswith('.csv') or n.endswith('.csv.gz') for n in js['partial_outputs']), js['partial_outputs']
    rc, out = run_cli(w, 'run_boot_big', extra=('--n-reps', str(R_REPS + 1)))
    assert rc == 1
    js = json.load(open(os.path.join(out, 'FAILED.json')))
    assert js['gate'] == 'bootstrap' and js['gates_passed_before_failure'][-1] == 'draws_binding'
    assert not os.path.exists(os.path.join(out, 'COMPLETE.json'))
    # gates-only runs no bootstrap, so the replicate count does not apply there
    rc, out = run_cli(w, 'run_boot_go', extra=('--gates-only', '--n-reps', str(R_REPS - 1)))
    assert rc == 0 and os.path.exists(os.path.join(out, 'GATES_ONLY.json'))


def test_bootstrap_evaluated_counter_detects_short_replicates(world, monkeypatch):
    """IR5-4: the every-combination counter counts actual replicate lengths, so a short replicate array fails."""
    orig = M.cell_statistics

    def short(*a, **kw):
        st = orig(*a, **kw)
        s = st[('reversed', 3, 'oracle')]['macro_recall']
        s.reps = s.reps[:-1]
        return st

    monkeypatch.setattr(M, 'cell_statistics', short)
    js, out = assert_failed(world, 'run_short', 'bootstrap')
    assert js['gates_passed_before_failure'][-1] == 'fig2c_identification'
    assert 'replicates for every overall metric' in js['message']
    assert 'table_s3_revised.csv' not in js['partial_outputs']


def test_post_bootstrap_oracle_table_and_k1_identity_checks(world, monkeypatch):
    """IR5-3: the unrounded ORACLE table values of order-invariant submitted rows must equal the submitted JSON
    (<= 1e-9); micro plurality@1 must equal micro hit@1. Failures are reported under the final bootstrap gate."""
    orig_cm = M.class_metrics

    def drift(tp, p, s):
        prec, rec, f1 = orig_cm(tp, p, s)
        return prec, rec, f1 + 1e-7

    monkeypatch.setattr(M, 'class_metrics', drift)
    js, out = assert_failed(world, 'run_post_cm', 'bootstrap')
    assert js['message'].startswith('post-bootstrap'), js['message']
    assert js['gates_passed_before_failure'][-1] == 'fig2c_identification'
    n_inv = len(world['spec']['expected_partition']['invariant'])
    assert sum('oracle canonical macro_f1' in d for d in js['detail']) == n_inv
    assert not any('micro_f1' in d or 'macro_recall' in d for d in js['detail'])
    assert 'table_s3_revised.csv' not in js['partial_outputs']
    monkeypatch.setattr(M, 'class_metrics', orig_cm)
    orig_st = M.cell_statistics

    def hit_off(*a, **kw):
        st = orig_st(*a, **kw)
        st[('canonical', 1, 'hp')]['hit_micro'].point += 1e-12
        return st

    monkeypatch.setattr(M, 'cell_statistics', hit_off)
    js, out = assert_failed(world, 'run_post_k1', 'bootstrap')
    assert js['detail'] and all(d.endswith('canonical: micro plurality@1 != micro hit@1') for d in js['detail'])
    assert len(js['detail']) == len(world['spec']['cells'])


def _complete_copy(w, name):
    rc, out = run_cli(w, name)
    assert rc == 0
    return out


def test_written_table_checks_are_exact(world, tmp_path):
    """IR5-3 written-table identity and IR5-5 exact 6-dp main-configuration comparison."""
    import shutil
    out = _complete_copy(world, 'run_exact')
    js = json.load(open(os.path.join(out, 'main_config_values.json')))
    M.check_written_k1_identity(out)
    M.check_main_config_tables(out, js)
    # a one-unit change at the 6th decimal of a k = 1 hit_micro value breaks the written identity
    bad = str(tmp_path / 'bad_k1')
    shutil.copytree(out, bad)
    hp = pd.read_csv(os.path.join(bad, 'hit_precision.csv'), dtype=str, keep_default_na=False)
    i = hp.index[(hp.k == '1') & (hp.metric == 'hit_micro')][0]
    hp.loc[i, 'value'] = f'{float(hp.loc[i, "value"]) + 1e-6:.6f}'
    hp.to_csv(os.path.join(bad, 'hit_precision.csv'), index=False, lineterminator='\n')
    with pytest.raises(M.GateError) as e:
        M.check_written_k1_identity(bad)
    assert e.value.gate == 'bootstrap'
    # main_config_values.json must equal the table strings exactly; 1e-6 at one CI bound fails
    for key, src, col in (('rule_independent@5:hit_macro', 'hit_precision.csv', 'ci_low'),
                          ('plurality@5:macro_f1', 'table_s3_revised.csv', 'value')):
        bad = str(tmp_path / f'bad_mc_{col}')
        shutil.copytree(out, bad)
        js2 = copy.deepcopy(js)
        v = js2['cells']['uni2/ot_cosine'][key]
        v[col] = round(v[col] + 1e-6, 6)
        open(os.path.join(bad, 'main_config_values.json'), 'w').write(json.dumps(js2))
        with pytest.raises(M.GateError) as e:
            M.check_main_config_tables(bad, js2)
        assert e.value.gate == 'main_config_crosscheck'


def test_k_columns_are_integers_and_winner_count_flag(world, capsys):
    """IR5-1: every k (and n_tied_winners) cell of every written CSV is an integer or blank; IR5-6: the ORACLE
    winner-count reproduction flag is in COMPLETE.json and the run log."""
    import re
    out = _complete_copy(world, 'run_kint')
    log = capsys.readouterr().out
    pat = re.compile(r'^\d*$')
    seen = set()
    for name in sorted(os.listdir(out)):
        if not (name.endswith('.csv') or name.endswith('.csv.gz')):
            continue
        df = pd.read_csv(os.path.join(out, name), dtype=str, keep_default_na=False)
        for col in ('k', 'n_tied_winners'):
            if col in df.columns:
                bad = [v for v in df[col].unique() if not pat.match(v)]
                assert not bad, (name, col, bad[:5])
                seen.add((name, col))
        if 'k' in df.columns and name != 'winners.csv':
            assert (df['k'] != '').all(), name
    assert ('winners.csv', 'k') in seen and ('winners.csv', 'n_tied_winners') in seen
    win = pd.read_csv(os.path.join(out, 'winners.csv'), dtype=str, keep_default_na=False)
    slot = win[win.row_type == 'slot']
    assert len(slot) and set(slot.k) == {'1', '3', '5', '10'} and (slot.n_tied_winners != '').all()
    comp = json.load(open(os.path.join(out, 'COMPLETE.json')))
    rec = json.load(open(os.path.join(out, 'reconciliation.json')))
    flag = rec['winner_count_reproduction']['equal']
    assert isinstance(flag, bool) and comp['oracle_winner_count_reproduces_submitted'] is flag
    assert f'[diagnostic] ORACLE winner count reproduces submitted count: {flag}' in log


def test_exact_winner_ties_are_split(tmp_path):
    w = make_world(str(tmp_path), dup={'uni2/ot_cosine': 'uni2/median_pooling', 'uni2/sampler': 'uni2/median_pooling'})
    rc, out = run_cli(w, 'run_tie')
    assert rc == 0
    win = pd.read_csv(os.path.join(out, 'winners.csv'))
    slot = win[(win.row_type == 'slot') & (win.slot_set == 'C32') & (win.display_encoder == 'UNIv2')]
    tied = slot[slot.n_tied_winners == 3]
    assert len(tied) > 0
    assert np.allclose(tied.weight, 1 / 3, atol=1e-6)
    assert set(tied.winner_cell) == {'uni2/median_pooling', 'uni2/ot_cosine', 'uni2/sampler'}
    # every (rule, order, k) slot contributes total weight 1 to the method counts
    cnt = win[win.row_type == 'count']
    for (sset, rule, order), g in cnt.groupby(['slot_set', 'rule', 'tie_order']):
        n_slots = {'C32': 8, 'H36': 12}.get(sset, 2 if sset.startswith('C32@') else 3)
        assert abs(g.weight.sum() - n_slots) < 1e-5, (sset, rule, order)   # weights are written at 6 dp
    # panel ranks: tied cells share the rank
    pan = pd.read_csv(os.path.join(out, 'fig2a_panel.csv'))
    for _, r in tied.drop_duplicates(['k', 'rule', 'tie_order']).iterrows():
        p = pan[(pan.slot_set == 'C32') & (pan.display_encoder == 'UNIv2') & (pan.k == r.k) & (pan.rule == r.rule)
                & (pan.tie_order == r.tie_order) & pan.is_winner]
        assert len(p) == 3 and (p['rank'] == 1).all()


def test_thread_limits_and_resources_recorded(world):
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ.get(v) == '1'
    rc, out = run_cli(world, 'run_res', extra=('--gates-only',))
    assert rc == 0
    rr = json.load(open(os.path.join(out, 'GATES_ONLY.json')))['resources']
    assert {'wall_seconds', 'cpu_seconds', 'peak_rss_mib'} <= set(rr)
