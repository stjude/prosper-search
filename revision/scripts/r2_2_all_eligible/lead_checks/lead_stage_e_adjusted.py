"""Analyst: Bonferroni-adjusted (12 primary contrasts -> 99.58% percentile) paired CIs for Yottixel, from the same
whole-patient bootstrap draws as lead_stage_e_verify.py; also the bootstrap fraction of draws with difference <= 0."""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
RUN = Path(sys.argv[1]); H = Path('/PROSPER_WORKSPACE')
meta = pd.read_csv(H/'comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv', dtype=str)
meta = meta[meta['Tumor.group']!='MISC'].reset_index(drop=True)
ids = meta['Slide.Scan.File'].tolist(); lab = meta['Tumor.group'].to_numpy(); pat = meta['UID.Subject'].to_numpy(); n=len(ids)
CLASSES = sorted(set(lab)); same = pat[:,None]==pat[None,:]
def flags(D):
    D=D.copy(); D[same]=np.inf; o=np.lexsort((np.tile(np.arange(n),(n,1)),D),axis=1)[:,:5]; nl=lab[o]
    return (nl==lab[:,None]).any(1).astype(float), (nl==lab[:,None]).mean(1)
def macro(x,w): return float(np.mean([np.average(x[lab==c],weights=w[lab==c]) for c in CLASSES if w[lab==c].sum()>0]))
z=np.load(RUN/'stage_d/yottixel/matrix/yottixel_matrix.npz'); zid=list(z['ids']); idx=[zid.index(s) for s in ids]; Da=z['d'][np.ix_(idx,idx)].astype(float)
Dm=pickle.load(open(H/'comet-feature-gen/comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_yottixel_no_same_patient.pkl','rb')).loc[ids,ids].to_numpy(float)
patients=tuple(sorted(set(pat))); P=len(patients); B=5000
rng=np.random.Generator(np.random.PCG64(123)); W=np.zeros((B,P),np.int64)
for b in range(B): W[b]=np.bincount(rng.integers(P,size=P),minlength=P)
W=W[:,pd.Index(patients).get_indexer(pat)]
fa=flags(Da); fm=flags(Dm); alpha_adj=0.05/12
print(f'Bonferroni over 12 primary contrasts: per-contrast alpha {alpha_adj:.5f} -> {100*(1-alpha_adj):.2f}% percentile CI')
for met,j in (('hit@5',0),('P@5',1)):
    xa,xm=fa[j],fm[j]
    dmi=(W@xa)/W.sum(1)-(W@xm)/W.sum(1); dma=np.array([macro(xa,W[b])-macro(xm,W[b]) for b in range(B)])
    for avg,dd,pt in (('micro',dmi,xa.mean()-xm.mean()),('macro',dma,macro(xa,np.ones(n))-macro(xm,np.ones(n)))):
        lo,hi=np.percentile(dd,[2.5,97.5]); alo,ahi=np.percentile(dd,[100*alpha_adj/2,100*(1-alpha_adj/2)])
        print(f'Yottixel {met:5s} {avg}: point {pt:+.4f}; 95% [{lo:+.4f},{hi:+.4f}]; adjusted [{alo:+.4f},{ahi:+.4f}]; '
              f'bootstrap draws with diff<=0: {(dd<=0).mean()*100:.2f}%')
