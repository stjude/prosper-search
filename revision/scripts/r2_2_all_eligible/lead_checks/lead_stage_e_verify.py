"""Analyst distinct-route recomputation of R2.2-ALL Stage E primary endpoints (hit@5 / precision@5, micro/macro,
whole-patient paired bootstrap) from raw Stage D matrices + published mosaic matrices + metadata. Own masking,
ranking, metrics and bootstrap; compared with stage_e/summary.csv and paired_differences.csv."""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
RUN = Path(sys.argv[1]); H = Path('/PROSPER_WORKSPACE')
META = H/'comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv'
PUB = H/'comet-feature-gen/comet-uni2/output/similarity_matrix'
CLASSES = ["ACT/ACC","DSRCT","EWS","LIVER","MEL","MPNST","MRT","NB","NF","NORMAL","OS","RB","RMS","SS","WT"]
meta = pd.read_csv(META, dtype=str)
meta = meta[meta['Tumor.group'] != 'MISC'].reset_index(drop=True)   # metadata order preserved
ids = meta['Slide.Scan.File'].tolist(); lab = meta['Tumor.group'].to_numpy(); pat = meta['UID.Subject'].to_numpy()
n = len(ids); print('population', n, 'patients', len(set(pat)), 'classes', len(set(lab)))
pos = {s:i for i,s in enumerate(ids)}
same = (pat[:,None] == pat[None,:])   # includes self

def topk_flags(D, k=5):
    """D aligned to metadata order. Mask self/same-patient with +inf; stable lexsort (distance, metadata pos)."""
    D = D.copy(); D[same] = np.inf
    assert np.isfinite(D[~same]).all() and (D[~same] < 1e6).all(), 'non-finite or sentinel among admissible cells'
    order = np.lexsort((np.tile(np.arange(n), (n,1)), D), axis=1)[:, :k]
    nl = lab[order]                       # (n,k) neighbour labels
    hit = (nl == lab[:,None]).any(1).astype(float)
    prec = (nl == lab[:,None]).mean(1)
    return hit, prec

def macro(x, w=None):
    w = np.ones(n) if w is None else w
    vals = []
    for c in CLASSES:
        m = lab == c
        if w[m].sum() == 0: continue
        vals.append(np.average(x[m], weights=w[m]))
    return float(np.mean(vals))

def load_all(arm):
    p = {'sampler': RUN/'stage_d/sampler/sampler_matrix.npz', 'median_pooling': RUN/'stage_d/median_pooling/median_pooling_matrix.npz',
         'yottixel': RUN/'stage_d/yottixel/matrix/yottixel_matrix.npz'}[arm]
    z = np.load(p); zid = list(z['ids']); idx = [zid.index(s) for s in ids]
    return z['d'][np.ix_(idx, idx)].astype(float)

def load_pub(arm):
    df = pickle.load(open(PUB/f'combined_similarity_matrix_uni2_{arm}_no_same_patient.pkl','rb'))
    return df.loc[ids, ids].to_numpy(dtype=float)

summ = pd.read_csv(RUN/'stage_e/summary.csv'); pdiff = pd.read_csv(RUN/'stage_e/paired_differences.csv')
# accepted bootstrap draws (definition copied: sorted unique patients, PCG64(123), multinomial via bincount)
patients = tuple(sorted(set(pat))); P = len(patients); B = 5000
rng = np.random.Generator(np.random.PCG64(123)); mult = np.zeros((B, P), np.int64)
for b in range(B): mult[b] = np.bincount(rng.integers(P, size=P), minlength=P)
pidx = pd.Index(patients).get_indexer(pat); W = mult[:, pidx]   # (B, n) per-slide weights

rows = []
for arm in ('sampler','median_pooling','yottixel'):
    res = {}
    for src, D in (('mosaic15', load_pub(arm)), ('all_eligible', load_all(arm))):
        hit, prec = topk_flags(D)
        res[src] = (hit, prec)
        tag = f"{arm}_{'mosaic' if src=='mosaic15' else 'all'}"
        for met, x in (('hit5', hit), ('precision5', prec)):
            for avg, v in (('micro', x.mean()), ('macro', macro(x))):
                ref = summ[(summ.arm==tag)&(summ.metric==met)&(summ.average==avg)].point.iloc[0]
                rows.append((tag, met, avg, round(v,6), round(ref,6), abs(v-ref) < 1e-9))
    # paired bootstrap all - mosaic
    for met, j in (('hit5',0), ('precision5',1)):
        xa, xm = res['all_eligible'][j], res['mosaic15'][j]
        d_micro = (W @ xa)/W.sum(1) - (W @ xm)/W.sum(1)
        d_macro = np.array([macro(xa, W[b]) - macro(xm, W[b]) for b in range(B)])
        for avg, dd, pt in (('micro', d_micro, xa.mean()-xm.mean()), ('macro', d_macro, macro(xa)-macro(xm))):
            lo, hi = np.percentile(dd, [2.5, 97.5])
            r = pdiff[(pdiff.minuend==f'{arm}_all')&(pdiff.metric==met)&(pdiff.average==avg)].iloc[0]
            rows.append((f'{arm} paired', met, avg, f'{pt:+.6f} [{lo:+.6f},{hi:+.6f}]', f'{r.point:+.6f} [{r.ci_lo:+.6f},{r.ci_hi:+.6f}]',
                         abs(pt-r.point)<1e-9 and abs(lo-r.ci_lo)<2e-3 and abs(hi-r.ci_hi)<2e-3))
out = pd.DataFrame(rows, columns=['arm','metric','avg','lead','implementer','agree'])
pd.set_option('display.width', 200); print(out.to_string(index=False)); print('ALL AGREE:', out.agree.all())
