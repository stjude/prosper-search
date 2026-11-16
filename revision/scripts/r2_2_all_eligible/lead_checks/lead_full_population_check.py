"""Analyst distinct-route full-population check of the R2.2-ALL bank and matrices.

Independent of the analyst's assembly/closeout code. Reads only: Stage A manifests, Stage B bank
(xy/coverage/source arrays; feats only for the pair checks), B3 json records, config_frozen.json,
the published mosaic matrices, the Stage D matrices, and the production prosper-search code.

Checks
  A  population: bank slides == non-MISC metadata slides == ids of all three Stage D matrices
  B  every slide: bank rows == manifest eligible rows (count, coordinate order, coverage); source codes
     0/1/2 == manifest mosaic_bank/full_bank/missing; eligible == in_cluster_locs; coverage >= 0.7 on the
     2,775 non-fallback slides; every mosaic tile present in the bank; no duplicate coordinates
  C  every B3 json record: GPU model A100-SXM4-80GB, config_sha256 == frozen, env_mismatch empty,
     n_tiles == bank code-2 count for that slide
  D  distance definitions/conventions: for seeded random pairs, published mosaic cells == production
     _pool(tmp_final rows) + cityblock (SAMPLER, median) and == hamming_min_median(barcodes_i, barcodes_j)
     (Yottixel; row = query); Stage D cells == the same from the bank rows
Usage: python3 lead_full_population_check.py RUN_DIR OUT_JSON [--workers 8]
"""
import sys, json, gzip, pickle, argparse, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cityblock
sys.path.insert(0, '/PROSPER_WORKSPACE/prosper-search')
from prosper.embeddings.aggregate_cli import _pool
from prosper.embeddings.slide_embedding.bob_embedding import hamming_min_median, features_to_barcodes

H = Path('/PROSPER_WORKSPACE')
META = H/'comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv'
PUB = H/'comet-feature-gen/comet-uni2/output/similarity_matrix'
TMP_FINAL = H/'comet-feature-gen/comet-uni2/data/tmp_final'
FALLBACK = {'9934', '11682', '14191'}
CODE = {'mosaic_bank': 0, 'full_bank': 1, 'missing': 2}

def check_slide(args):
    run, sid = args; run = Path(run)
    mdir = run/'stage_a'/('slides_option_i' if sid in FALLBACK else 'slides')
    m = pd.read_csv(mdir/f'{sid}.manifest.csv.gz', float_precision='round_trip', keep_default_na=False)
    e = m[m['eligible'] == 1]
    z = np.load(run/'stage_b/bank'/f'{sid}.npz')
    xy, cov, src = z['xy'], z['coverage'], z['source']
    r = {'slide': sid, 'n_bank': int(len(src)), 'n_eligible': int(len(e)),
         'n_mosaic_manifest': int((m['in_mosaic'] == 1).sum()),
         'code_counts': [int((src == c).sum()) for c in (0, 1, 2)]}
    ok = {}
    ok['rows_equal'] = len(src) == len(e)
    ok['coords_order_equal'] = ok['rows_equal'] and bool(np.array_equal(xy[:, 0], e['wsi_loc_x'].to_numpy()) and np.array_equal(xy[:, 1], e['wsi_loc_y'].to_numpy()))
    ok['coverage_equal'] = ok['rows_equal'] and bool(np.allclose(cov.astype(float), e['coverage'].to_numpy(float), atol=1e-9))
    ok['codes_equal'] = ok['rows_equal'] and bool(np.array_equal(src.astype(int), e['source'].map(CODE).to_numpy()))
    ok['eligible_is_cluster_locs'] = bool((m['eligible'] == m['in_cluster_locs']).all())
    ok['coverage_ge_0.7'] = True if sid in FALLBACK else bool((e['coverage'].to_numpy(float) >= 0.7).all())
    ok['mosaic_all_in_bank'] = bool(((m['in_mosaic'] == 1) <= (m['eligible'] == 1)).all()) and int(((m['in_mosaic'] == 1) & (m['eligible'] == 1)).sum()) == r['code_counts'][0]
    ok['no_dup_coords'] = len({(int(a), int(b)) for a, b in xy}) == len(xy)
    ok['mosaic_rows_have_code0'] = ok['rows_equal'] and bool((src[e['in_mosaic'].to_numpy() == 1] == 0).all())
    r['ok'] = ok; r['all_ok'] = all(ok.values())
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); run = Path(a.run); out = {}
    meta = pd.read_csv(META, dtype=str); meta = meta[meta['Tumor.group'] != 'MISC']
    ids = sorted(meta['Slide.Scan.File']); bank = sorted(p.stem for p in (run/'stage_b/bank').glob('*.npz'))
    mats = {'sampler': run/'stage_d/sampler/sampler_matrix.npz', 'median_pooling': run/'stage_d/median_pooling/median_pooling_matrix.npz', 'yottixel': run/'stage_d/yottixel/matrix/yottixel_matrix.npz'}
    mat_ids = {k: sorted(np.load(v)['ids'].tolist()) for k, v in mats.items()}
    out['A_population'] = {'n_meta_nonmisc': len(ids), 'n_bank': len(bank), 'bank_eq_meta': bank == ids, **{f'{k}_ids_eq_meta': v == ids for k, v in mat_ids.items()}}
    print('A', out['A_population'], flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(check_slide, [(str(run), s) for s in ids], chunksize=16))
    bad = [r for r in res if not r['all_ok']]
    out['B_per_slide'] = {'n_checked': len(res), 'n_all_ok': len(res) - len(bad), 'failures': bad[:20],
                          'total_rows': int(sum(r['n_bank'] for r in res)), 'total_eligible': int(sum(r['n_eligible'] for r in res)),
                          'total_codes': [int(sum(r['code_counts'][c] for r in res)) for c in (0, 1, 2)],
                          'total_mosaic_manifest': int(sum(r['n_mosaic_manifest'] for r in res)),
                          'checks_failed_by_name': {k: sum(1 for r in res if not r['ok'][k]) for k in res[0]['ok']}}
    print('B', {k: v for k, v in out['B_per_slide'].items() if k != 'failures'}, flush=True)
    frozen = json.load(open(run/'stage_b/config_frozen.json'))['config_sha256']; code2 = {r['slide']: r['code_counts'][2] for r in res}
    recs = list((run/'stage_b/b3/out').glob('*.json')) + list((run/'stage_b/b3_fallback/out').glob('*.json'))
    gpus, badc = {}, []
    for p in recs:
        j = json.load(open(p)); gpus[j['gpu_model']] = gpus.get(j['gpu_model'], 0) + 1
        if not (j['gpu_model'] == 'NVIDIA A100-SXM4-80GB' and j['config_sha256'] == frozen and not j['env_mismatch'] and j['n_tiles'] == code2.get(j['slide'])):
            badc.append(j['slide'])
    out['C_b3_records'] = {'n_records': len(recs), 'gpu_models': gpus, 'n_bad': len(badc), 'bad': badc[:20], 'n_slides_with_code2': sum(1 for v in code2.values() if v > 0),
                           'n_code2_rows_in_records': int(sum(json.load(open(p))['n_tiles'] for p in recs))}
    print('C', out['C_b3_records'], flush=True)
    rng = np.random.default_rng(2026); pairs = []
    pool = [s for s in ids if s not in FALLBACK]
    while len(pairs) < 4:
        i, j = rng.choice(pool, 2, replace=False); pairs.append((str(i), str(j)))
    pairs.append(('9934', str(rng.choice(pool))))   # one fallback slide
    pub = {k: pickle.load(open(PUB/f'combined_similarity_matrix_uni2_{k}_no_same_patient.pkl', 'rb')) for k in mats}
    sd = {k: np.load(v) for k, v in mats.items()}; sidx = {k: {s: n for n, s in enumerate(sd[k]['ids'].tolist())} for k in mats}
    D = []
    for i, j in pairs:
        fi = pickle.load(open(TMP_FINAL/f'{i}.pkl', 'rb'))[0]; fj = pickle.load(open(TMP_FINAL/f'{j}.pkl', 'rb'))[0]
        bi = np.load(run/'stage_b/bank'/f'{i}.npz')['feats']; bj = np.load(run/'stage_b/bank'/f'{j}.npz')['feats']
        row = {'pair': (i, j), 'n_mosaic': (fi.shape[0], fj.shape[0]), 'n_all': (bi.shape[0], bj.shape[0])}
        for k in ('sampler', 'median_pooling'):
            pm = cityblock(_pool(fi, k).astype(np.float64), _pool(fj, k).astype(np.float64)); pa = cityblock(_pool(bi, k).astype(np.float64), _pool(bj, k).astype(np.float64))
            row[k] = {'pub_cell': float(pub[k].loc[i, j]), 'lead_mosaic': float(pm), 'pub_match': bool(np.isclose(pm, pub[k].loc[i, j], rtol=1e-9, atol=1e-6)),
                      'stage_d_cell': float(sd[k]['d'][sidx[k][i], sidx[k][j]]), 'lead_all': float(pa), 'stage_d_match': bool(np.isclose(pa, sd[k]['d'][sidx[k][i], sidx[k][j]], rtol=1e-9, atol=1e-6))}
        hm_ij = hamming_min_median(features_to_barcodes(fi), features_to_barcodes(fj)); hm_ji = hamming_min_median(features_to_barcodes(fj), features_to_barcodes(fi))
        ha_ij = hamming_min_median(features_to_barcodes(bi), features_to_barcodes(bj)); ha_ji = hamming_min_median(features_to_barcodes(bj), features_to_barcodes(bi))
        y = pub['yottixel']; k = 'yottixel'
        row[k] = {'pub_ij': float(y.loc[i, j]), 'pub_ji': float(y.loc[j, i]), 'lead_mosaic_ij': float(hm_ij), 'lead_mosaic_ji': float(hm_ji),
                  'pub_match_both': bool(hm_ij == y.loc[i, j] and hm_ji == y.loc[j, i]),
                  'stage_d_ij': float(sd[k]['d'][sidx[k][i], sidx[k][j]]), 'stage_d_ji': float(sd[k]['d'][sidx[k][j], sidx[k][i]]), 'lead_all_ij': float(ha_ij), 'lead_all_ji': float(ha_ji),
                  'stage_d_match_both': bool(ha_ij == sd[k]['d'][sidx[k][i], sidx[k][j]] and ha_ji == sd[k]['d'][sidx[k][j], sidx[k][i]])}
        D.append(row); print('D', json.dumps(row), flush=True)
    out['D_pairs'] = D
    out['D_all_match'] = all(r[k][m] for r in D for k in ('sampler', 'median_pooling') for m in ('pub_match', 'stage_d_match')) and all(r['yottixel']['pub_match_both'] and r['yottixel']['stage_d_match_both'] for r in D)
    out['VERDICT'] = {'A': all(v for k, v in out['A_population'].items() if isinstance(v, bool)), 'B': len(bad) == 0, 'C': len(badc) == 0 and gpus == {'NVIDIA A100-SXM4-80GB': len(recs)}, 'D': out['D_all_match']}
    print('VERDICT', out['VERDICT'], flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=str)

if __name__ == '__main__':
    main()
