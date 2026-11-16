"""Analyst checks part 3: (b') Yottixel all-eligible matrix cells recomputed end-to-end from the Stage B bank
(bank feats -> production features_to_barcodes -> production CPU hamming_min_median), a route independent of
Stage C prep barcodes and the Stage D GPU rows; direction convention checked on the published mosaic matrix.
(c') corrected hit@5 tie-dependence rule (slots remaining after strictly-closer neighbours) + fullrow bounds."""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/PROSPER_WORKSPACE/prosper-search')
from prosper.embeddings.slide_embedding.bob_embedding import hamming_min_median, features_to_barcodes
RUN = Path(sys.argv[1]); H = Path('/PROSPER_WORKSPACE')
META = H/'comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv'
PUB = H/'comet-feature-gen/comet-uni2/output/similarity_matrix'
meta = pd.read_csv(META, dtype=str); meta = meta[meta['Tumor.group']!='MISC'].reset_index(drop=True)
ids = meta['Slide.Scan.File'].tolist(); lab = meta['Tumor.group'].to_numpy(); pat = meta['UID.Subject'].to_numpy(); n=len(ids)
same = pat[:,None]==pat[None,:]
z = np.load(RUN/'stage_d/yottixel/matrix/yottixel_matrix.npz'); zid=list(z['ids']); d=z['d']; exc=z['excluded']
Dm_df = pickle.load(open(PUB/'combined_similarity_matrix_uni2_yottixel_no_same_patient.pkl','rb'))
Dm_full = Dm_df.loc[zid, zid].to_numpy(float)
print('published mosaic yottixel symmetric?', np.allclose(Dm_full, Dm_full.T), '| stage_d yottixel symmetric?', np.allclose(d, d.T))
rng = np.random.default_rng(11); cells=[]
while len(cells)<6:
    i,j = rng.integers(len(zid), size=2)
    if i!=j and not exc[i,j]: cells.append((i,j))
allok=True
for i,j in cells:
    fa = np.load(RUN/'stage_b/bank'/f'{zid[i]}.npz')['feats']; fb = np.load(RUN/'stage_b/bank'/f'{zid[j]}.npz')['feats']
    A = features_to_barcodes(fa); B = features_to_barcodes(fb)
    ij = hamming_min_median(A,B); ji = hamming_min_median(B,A)
    ok = (ij==d[i,j]) and (ji==d[j,i]); allok &= ok
    print(f"(b') {zid[i]}({A.shape}) vs {zid[j]}({B.shape}): matrix d[i,j]={d[i,j]:.1f} d[j,i]={d[j,i]:.1f} | bank->prod CPU hmm(i,j)={ij:.1f} hmm(j,i)={ji:.1f} | match={ok}")
print("(b') all 6 pairs match both directions:", bool(allok))
# direction convention on mosaic: compare published cell with production-from-saved-mosaic? (only orientation check here)
idx=[zid.index(s) for s in ids]; Dall=d[np.ix_(idx,idx)].astype(float); Dall[same]=np.inf
Dm = Dm_df.loc[ids,ids].to_numpy(float); Dm[same]=np.inf
tb = pd.read_csv(RUN/'stage_e/tie_bounds.csv').set_index('arm')
for tag, D in (('yottixel_mosaic',Dm),('yottixel_all',Dall)):
    S = np.sort(D,axis=1); dep=0; lo=0.0; hi=0.0
    for q in range(n):
        v=S[q,4]; fixed = lab[D[q]<v]; block = lab[D[q]==v]; slots = 5-len(fixed)
        m = int((block==lab[q]).sum()); nm = len(block)-m
        fixed_hit = bool((fixed==lab[q]).any())
        best = fixed_hit or m>=1; worst = fixed_hit or nm < slots
        hi += best; lo += worst; dep += (best != worst)
    r = tb.loc[tag]
    print(f"(c') {tag}: hit5 tie-dependent lead {dep} vs impl {int(r.queries_hit5_tie_dependent_fullrow)}; micro hit5 bounds lead [{lo/n:.6f},{hi/n:.6f}] vs impl [{r.micro_hit5_fullrow_lower:.6f},{r.micro_hit5_fullrow_upper:.6f}]")
