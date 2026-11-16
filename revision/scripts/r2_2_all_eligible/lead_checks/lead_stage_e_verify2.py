"""Analyst distinct-route checks, part 2: (a) mosaic points vs accepted c2_metric_suite; (b) Yottixel all-eligible
matrix cells recomputed from Stage C prep barcodes with the production hamming_min_median (CPU) — distinct from the
GPU row route; (c) Yottixel tie-dependence counts (rank-5/6 tie, hit@5 tie-dependent) vs stage_e/tie_bounds.csv."""
import sys, json, pickle, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/PROSPER_WORKSPACE/prosper-search')
from prosper.embeddings.slide_embedding.bob_embedding import hamming_min_median
RUN = Path(sys.argv[1]); OUT = Path(sys.argv[2]); H = Path('/PROSPER_WORKSPACE')
META = H/'comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv'
PUB = H/'comet-feature-gen/comet-uni2/output/similarity_matrix'
meta = pd.read_csv(META, dtype=str); meta = meta[meta['Tumor.group']!='MISC'].reset_index(drop=True)
ids = meta['Slide.Scan.File'].tolist(); lab = meta['Tumor.group'].to_numpy(); pat = meta['UID.Subject'].to_numpy(); n=len(ids)
same = pat[:,None]==pat[None,:]
# (a) accepted mosaic metrics
ok=[]
for arm in ('sampler','median_pooling','yottixel'):
    pe = json.load(open(OUT/'c2_metric_suite'/f'metrics_uni2_{arm}.json'))['point_estimates']
    df = pickle.load(open(PUB/f'combined_similarity_matrix_uni2_{arm}_no_same_patient.pkl','rb')); D = df.loc[ids,ids].to_numpy(float); D[same]=np.inf
    order = np.lexsort((np.tile(np.arange(n),(n,1)), D), axis=1)[:,:5]; nl = lab[order]; hit=(nl==lab[:,None]).any(1); prec=(nl==lab[:,None]).mean(1)
    ok.append((arm, abs(hit.mean()-pe['micro_hit_at_5'])<1e-12, abs(prec.mean()-pe['micro_precision_at_5'])<1e-12))
print('(a) mosaic points vs accepted c2 json (micro hit5, micro P5):', ok)
# (b) Yottixel cells from barcodes
z = np.load(RUN/'stage_d/yottixel/matrix/yottixel_matrix.npz'); zid=list(z['ids']); d=z['d']; exc=z['excluded']
rng = np.random.default_rng(11); cells=[]
while len(cells)<6:
    i,j = rng.integers(len(zid), size=2)
    if i!=j and not exc[i,j]: cells.append((i,j))
for i,j in cells:
    A = np.load(RUN/'stage_c/prep/barcodes'/f'{zid[i]}.npy'); B = np.load(RUN/'stage_c/prep/barcodes'/f'{zid[j]}.npy')
    cpu_ij = hamming_min_median(A,B); cpu_ji = hamming_min_median(B,A)
    print(f'(b) {zid[i]}({A.shape[0]}) vs {zid[j]}({B.shape[0]}): matrix {d[i,j]:.3f} / {d[j,i]:.3f}; cpu ij {cpu_ij:.3f} ji {cpu_ji:.3f}; '
          f'sym-combined candidates mean {0.5*(cpu_ij+cpu_ji):.3f} min {min(cpu_ij,cpu_ji):.3f} max {max(cpu_ij,cpu_ji):.3f}')
# (c) Yottixel tie-dependence
idx=[zid.index(s) for s in ids]; Dall=d[np.ix_(idx,idx)].astype(float); Dall[same]=np.inf
Dm = pickle.load(open(PUB/'combined_similarity_matrix_uni2_yottixel_no_same_patient.pkl','rb')).loc[ids,ids].to_numpy(float); Dm[same]=np.inf
tb = pd.read_csv(RUN/'stage_e/tie_bounds.csv').set_index('arm')
for tag, D in (('yottixel_mosaic',Dm),('yottixel_all',Dall)):
    S = np.sort(D,axis=1); tie56 = (S[:,4]==S[:,5]).sum()
    # hit@5 tie-dependent: 5th-smallest value v; labels among cells with D<v fixed; cells == v form the tie block
    dep=0
    for q in range(n):
        v=S[q,4]; fixed = lab[D[q]<v]; block = lab[D[q]==v]
        if (fixed==lab[q]).any(): continue           # already a hit regardless
        if (block==lab[q]).any() and not (block==lab[q]).all(): dep+=1   # hit depends on tie resolution
    r = tb.loc[tag]
    print(f'(c) {tag}: rank5/6 ties lead {tie56} vs impl {r.queries_rank56_tie}; hit5 tie-dependent lead {dep} vs impl {r.queries_hit5_tie_dependent_fullrow}')
