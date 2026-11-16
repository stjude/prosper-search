"""Analyst precision-sensitivity check for R2.2-ALL.

Question: the all-eligible bank mixes fp32 rows (mosaic + newly extracted gap tiles) with fp16-autocast rows
reused from the tile-ladder full-tile bank (code 1, 79.5% of rows). Could that precision mix, rather than the
tile count, explain the all-eligible vs mosaic differences?

Design (same tile set, precision regime swapped): for every slide, the saved 15% mosaic rows (fp32, tmp_final)
are compared with the SAME mosaic coordinates taken from the fp16-autocast bank where present ("fp16-substituted
mosaic"; the 3,548,095 shared tiles). Then
  1. per-tile fp32-vs-fp16 agreement (cosine; barcode bit flips) over ALL shared tiles (distinct route to Stage A)
  2. pooling arms (SAMPLER, median): full 2,778x2,778 L1 matrices for fp32-mosaic and fp16sub-mosaic; hit@5 /
     precision@5 micro/macro; per-query top-5 changes; compared with the mosaic -> all-eligible effect
  3. Yottixel: N seeded query slides vs the full library under fp32 and fp16sub barcodes on both sides; top-5 and
     hit@5 changes; the fp32 rows must equal the published mosaic rows
Outputs under OUT_DIR (sibling of run_001; run_001 untouched). Usage: python3 lead_precision_sensitivity.py RUN OUT_DIR [--workers 16] [--yq 24]
"""
import sys, json, pickle, argparse, time, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, '/PROSPER_WORKSPACE/prosper-search')
from prosper.embeddings.aggregate_cli import _pool
from prosper.embeddings.slide_embedding.bob_embedding import hamming_min_median, features_to_barcodes

H = Path('/PROSPER_WORKSPACE')
META = H/'comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv'
PUB = H/'comet-feature-gen/comet-uni2/output/similarity_matrix'
TMP_FINAL = H/'comet-feature-gen/comet-uni2/data/tmp_final'
FP16 = H/'prosper_revision/prosper_revision_output/features/uni2_all_patches'
CLASSES = ["ACT/ACC","DSRCT","EWS","LIVER","MEL","MPNST","MRT","NB","NF","NORMAL","OS","RB","RMS","SS","WT"]

def one(args):
    sid, outdir = args; outdir = Path(outdir)
    f32, locs = pickle.load(open(TMP_FINAL/f'{sid}.pkl', 'rb')); f32 = np.asarray(f32, np.float32)
    coords = [tuple(int(v) for v in d['wsi_loc']) for d in locs]
    sub = f32.copy(); nshared = 0; cos = np.array([]); flips = np.array([])
    p16 = FP16/f'{sid}.pkl'
    if p16.exists():
        g16, locs16 = pickle.load(open(p16, 'rb')); g16 = np.asarray(g16, np.float32)
        idx16 = {tuple(int(v) for v in d['wsi_loc']): k for k, d in enumerate(locs16)}
        rows = [(r, idx16[c]) for r, c in enumerate(coords) if c in idx16]
        if rows:
            r32 = np.array([r for r, _ in rows]); r16 = np.array([k for _, k in rows]); nshared = len(rows)
            a, b = f32[r32], g16[r16]
            cos = (a*b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            flips = (features_to_barcodes(a) != features_to_barcodes(b)).sum(1)
            sub[r32] = b
    np.save(outdir/'bc32'/f'{sid}.npy', np.packbits(features_to_barcodes(f32), axis=1))
    np.save(outdir/'bc16'/f'{sid}.npy', np.packbits(features_to_barcodes(sub), axis=1))
    return {'slide': sid, 'n_mosaic': int(f32.shape[0]), 'n_shared': int(nshared), 'cos_min': float(cos.min()) if nshared else None,
            'cos_mean': float(cos.mean()) if nshared else None, 'flips_sum': int(flips.sum()) if nshared else 0,
            'flips_max': int(flips.max()) if nshared else 0, 'flips_hist': np.bincount(flips, minlength=12)[:12].tolist() if nshared else [0]*12,
            'v32_s': _pool(f32, 'sampler').astype(np.float64), 'v16_s': _pool(sub, 'sampler').astype(np.float64),
            'v32_m': _pool(f32, 'median_pooling').astype(np.float64), 'v16_m': _pool(sub, 'median_pooling').astype(np.float64)}

def yott_query(args):
    q, outdir = args; out = Path(outdir)
    ids, lab = pickle.load(open(out/'_meta.pkl', 'rb')); n = len(ids)
    same = np.load(out/'_same.npy'); Dall_y = np.load(out/'_dall_y.npy', mmap_mode='r'); puby = np.load(out/'_puby.npy', mmap_mode='r')
    def unpack(p): return np.unpackbits(np.load(p), axis=1, count=1535)
    q32 = unpack(out/'bc32'/f'{ids[q]}.npy'); q16 = unpack(out/'bc16'/f'{ids[q]}.npy')
    r32 = np.zeros(n); r16 = np.zeros(n)
    for j in range(n):
        if j == q or same[q, j]: continue
        r32[j] = hamming_min_median(q32, unpack(out/'bc32'/f'{ids[j]}.npy')); r16[j] = hamming_min_median(q16, unpack(out/'bc16'/f'{ids[j]}.npy'))
    pubrow = np.asarray(puby[q]); ok = bool(np.array_equal(r32[~same[q]], pubrow[~same[q]]))
    def top5(row):
        rr = np.array(row, dtype=float); rr[same[q]] = np.inf; return np.lexsort((np.arange(n), rr))[:5]
    t32, t16, tall = top5(r32), top5(r16), top5(Dall_y[q])
    return {'query': ids[q], 'label': lab[q], 'fp32_row_equals_published': ok, 'max_abs_row_change_fp16sub': float(np.abs(r16-r32)[~same[q]].max()),
            'median_abs_row_change_fp16sub': float(np.median(np.abs(r16-r32)[~same[q]])), 'median_abs_row_change_all': float(np.median(np.abs(np.asarray(Dall_y[q])-r32)[~same[q]])),
            'top5_set_changed_fp16sub': bool(set(t32) != set(t16)), 'top5_set_changed_all': bool(set(t32) != set(tall)),
            'hit5_fp32': bool((lab[t32] == lab[q]).any()), 'hit5_fp16sub': bool((lab[t16] == lab[q]).any()), 'hit5_all': bool((lab[tall] == lab[q]).any()),
            'p5_fp32': float((lab[t32] == lab[q]).mean()), 'p5_fp16sub': float((lab[t16] == lab[q]).mean()), 'p5_all': float((lab[tall] == lab[q]).mean())}

def metrics(D, lab, same):
    n = len(lab); D = D.copy(); D[same] = np.inf
    o = np.lexsort((np.tile(np.arange(n), (n, 1)), D), axis=1)[:, :5]; nl = lab[o]
    hit = (nl == lab[:, None]).any(1).astype(float); prec = (nl == lab[:, None]).mean(1)
    mac = lambda x: float(np.mean([x[lab == c].mean() for c in CLASSES]))
    return {'hit5_micro': float(hit.mean()), 'hit5_macro': mac(hit), 'p5_micro': float(prec.mean()), 'p5_macro': mac(prec)}, o, hit, prec

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--yq', type=int, default=24)
    a = ap.parse_args(); run = Path(a.run); out = Path(a.out); (out/'bc32').mkdir(parents=True, exist_ok=True); (out/'bc16').mkdir(exist_ok=True)
    meta = pd.read_csv(META, dtype=str); meta = meta[meta['Tumor.group'] != 'MISC'].reset_index(drop=True)
    ids = meta['Slide.Scan.File'].tolist(); lab = meta['Tumor.group'].to_numpy(); pat = meta['UID.Subject'].to_numpy(); n = len(ids)
    same = pat[:, None] == pat[None, :]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, [(s, str(out)) for s in ids], chunksize=8))
    print(f'phase 1 done in {time.time()-t0:.0f}s', flush=True)
    rep = {'n_slides': n, 'n_slides_with_fp16': sum(1 for r in res if r['n_shared'] > 0), 'n_shared_tiles': int(sum(r['n_shared'] for r in res)),
           'n_mosaic_tiles': int(sum(r['n_mosaic'] for r in res)),
           'cos_min_over_all_shared': min(r['cos_min'] for r in res if r['n_shared']),
           'cos_mean_tilew': float(sum(r['cos_mean']*r['n_shared'] for r in res if r['n_shared']) / sum(r['n_shared'] for r in res)),
           'barcode_flips_per_tile_mean': float(sum(r['flips_sum'] for r in res) / sum(r['n_shared'] for r in res)),
           'barcode_flips_per_tile_max': max(r['flips_max'] for r in res),
           'barcode_flips_hist_0_11plus': np.sum([r['flips_hist'] for r in res], axis=0).tolist(),
           'per_bit_flip_rate': float(sum(r['flips_sum'] for r in res) / (sum(r['n_shared'] for r in res) * 1535))}
    print('1 per-tile:', json.dumps(rep), flush=True)
    # 2. pooling arms
    pub = {k: pickle.load(open(PUB/f'combined_similarity_matrix_uni2_{k}_no_same_patient.pkl', 'rb')).loc[ids, ids].to_numpy(float) for k in ('sampler', 'median_pooling', 'yottixel')}
    allm = {'sampler': run/'stage_d/sampler/sampler_matrix.npz', 'median_pooling': run/'stage_d/median_pooling/median_pooling_matrix.npz', 'yottixel': run/'stage_d/yottixel/matrix/yottixel_matrix.npz'}
    def load_all(k):
        z = np.load(allm[k]); zid = list(z['ids']); idx = [zid.index(s) for s in ids]; return z['d'][np.ix_(idx, idx)].astype(float)
    rep['pooling'] = {}
    for k, key in (('sampler', 's'), ('median_pooling', 'm')):
        V32 = np.stack([r[f'v32_{key}'] for r in res]); V16 = np.stack([r[f'v16_{key}'] for r in res])
        np.savez_compressed(out/f'pooled_{k}.npz', ids=np.array(ids), fp32_mosaic=V32, fp16sub_mosaic=V16)
        D32 = squareform(pdist(V32, 'cityblock')); D16 = squareform(pdist(V16, 'cityblock')); Dall = load_all(k)
        offd = ~same
        m32, o32, h32, p32 = metrics(D32, lab, same); m16, o16, h16, p16_ = metrics(D16, lab, same); mall, oall, hall, pall = metrics(pub[k], lab, same)
        mall2, oall2, hall2, pall2 = metrics(Dall, lab, same)
        dprec = np.abs(V32 - V16).sum(1)
        rep['pooling'][k] = {
            'fp32_mosaic_equals_published': bool(np.allclose(D32[offd], pub[k][offd], rtol=1e-9, atol=1e-6)),
            'metrics_fp32_mosaic': m32, 'metrics_fp16sub_mosaic': m16, 'metrics_all_eligible': mall2,
            'delta_fp16sub_minus_fp32': {kk: m16[kk]-m32[kk] for kk in m32}, 'delta_all_minus_fp32': {kk: mall2[kk]-m32[kk] for kk in m32},
            'queries_top5_set_changed_fp16sub': int(sum(set(o32[q]) != set(o16[q]) for q in range(n))),
            'queries_top5_set_changed_all': int(sum(set(o32[q]) != set(oall2[q]) for q in range(n))),
            'queries_hit5_changed_fp16sub': int((h32 != h16).sum()), 'queries_hit5_changed_all': int((h32 != hall2).sum()),
            'median_rel_cell_change_fp16sub': float(np.median(np.abs(D16[offd]-D32[offd]) / D32[offd])),
            'median_rel_cell_change_all': float(np.median(np.abs(Dall[offd]-D32[offd]) / D32[offd])),
            'vector_L1_shift_fp16sub_median': float(np.median(dprec)), 'vector_L1_shift_fp16sub_max': float(dprec.max()),
            'nn1_distance_fp32_median': float(np.median(np.where(same, np.inf, D32).min(1)))}
        print(f'2 {k}:', json.dumps(rep['pooling'][k]), flush=True)
    # 3. Yottixel queries (one process per query; each process reads the cached packed barcodes)
    rng = np.random.default_rng(99); qs = [int(q) for q in rng.choice(n, a.yq, replace=False)]
    Dall_y = load_all('yottixel'); np.save(out/'_dall_y.npy', Dall_y); np.save(out/'_puby.npy', pub['yottixel']); np.save(out/'_same.npy', same)
    pickle.dump((ids, lab), open(out/'_meta.pkl', 'wb'))
    t1 = time.time(); yrows = []
    with ProcessPoolExecutor(min(a.yq, a.workers)) as ex:
        for row in ex.map(yott_query, [(q, str(out)) for q in qs]):
            yrows.append(row); print('3', json.dumps(row), f'{time.time()-t1:.0f}s', flush=True)
    yd = pd.DataFrame(yrows)
    rep['yottixel'] = {'n_queries': int(len(yd)), 'all_fp32_rows_equal_published': bool(yd.fp32_row_equals_published.all()),
                       'max_abs_row_change_fp16sub': float(yd.max_abs_row_change_fp16sub.max()), 'median_abs_row_change_fp16sub': float(yd.median_abs_row_change_fp16sub.median()),
                       'median_abs_row_change_all': float(yd.median_abs_row_change_all.median()),
                       'queries_top5_changed_fp16sub': int(yd.top5_set_changed_fp16sub.sum()), 'queries_top5_changed_all': int(yd.top5_set_changed_all.sum()),
                       'queries_hit5_changed_fp16sub': int((yd.hit5_fp32 != yd.hit5_fp16sub).sum()), 'queries_hit5_changed_all': int((yd.hit5_fp32 != yd.hit5_all).sum()),
                       'mean_p5_fp32': float(yd.p5_fp32.mean()), 'mean_p5_fp16sub': float(yd.p5_fp16sub.mean()), 'mean_p5_all': float(yd.p5_all.mean())}
    yd.to_csv(out/'yottixel_queries.csv', index=False)
    print('3 yottixel:', json.dumps(rep['yottixel']), flush=True)
    json.dump(rep, open(out/'precision_sensitivity.json', 'w'), indent=1, default=str)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('v')} for r in res]).to_csv(out/'per_slide.csv', index=False)

if __name__ == '__main__':
    main()
