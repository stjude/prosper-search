"""Analyst distinct-route verification of the R2.2-ALL Stage B assembled bank (seed 7, 15 slides; v2 — fixed exact-name sha lookup, config_sha256 key, hoisted npz arrays)."""
import sys, json, pickle, random, hashlib, time, gzip
import numpy as np, pandas as pd
from pathlib import Path
RUN = Path(sys.argv[1]); SB = RUN/'stage_b'; H = Path('/PROSPER_WORKSPACE')
FB = {'9934','11682','14191'}
ps = pd.read_csv(SB/'bank_per_slide.csv', dtype={'slide':str})
print('per-slide csv: n=%d rows=%d c0=%d c1=%d c2=%d all_match=%s' % (len(ps), ps.n_rows.sum(), ps.n_code0.sum(), ps.n_code1.sum(), ps.n_code2.sum(), ps.matches_gap_d1.all()))
assert (ps.n_rows == ps.n_code0+ps.n_code1+ps.n_code2).all()
assert set(ps.b3_gpu.dropna()) <= {'NVIDIA A100-SXM4-80GB'}, set(ps.b3_gpu.dropna())
# Stage A aggregate cross-check (independent of gap_d1)
sha_lines = dict(l.split()[::-1] for l in (SB/'bank_SHA256SUMS.txt').read_text().splitlines() if l.strip())
print('bank_SHA256SUMS entries', len(sha_lines))
random.seed(7)
samp = random.sample(sorted(set(ps.slide)-FB), 12) + sorted(FB)
code_of = {'mosaic_bank':0, 'full_bank':1, 'missing':2}
out = []
for s in samp:
    t0=time.time()
    npz = SB/'bank'/f'{s}.npz'
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    key = next(k for k in sha_lines if k.rsplit('/',1)[-1]==f'{s}.npz')
    ok_sha = sha_lines[key] == sha and ps.set_index('slide').loc[s,'npz_sha256'] == sha
    z = np.load(npz); f, xy, cov, src = z['feats'], z['xy'], z['coverage'], z['source']
    assert f.dtype==np.float32 and f.shape[1]==1536 and np.isfinite(f).all()
    keys = [tuple(map(int,p)) for p in xy]
    ok_dup = len(set(keys)) == len(keys)
    m = pd.read_csv(RUN/'stage_a'/('slides_option_i' if s in FB else 'slides')/f'{s}.manifest.csv.gz', float_precision='round_trip')
    me = m[m.eligible==1]
    ok_n = len(me) == len(f) == ps.set_index('slide').loc[s,'n_rows']
    mk = {(int(r.wsi_loc_x),int(r.wsi_loc_y)):(code_of[r.source], float(r.coverage)) for r in me.itertuples()}
    ok_set = set(mk) == set(keys)
    # per-row source code and coverage vs manifest
    ok_code = all(mk[k][0]==int(c) for k,c in zip(keys,src))
    ok_cov = all(abs(mk[k][1]-float(c))<1e-9 for k,c in zip(keys,cov))
    # byte equality per code
    mf, ml = pickle.load(open(H/'comet-feature-gen/comet-uni2/data/tmp_final'/f'{s}.pkl','rb'))
    mpos = {tuple(map(int,d['wsi_loc'])):i for i,d in enumerate(ml)}
    i0 = np.flatnonzero(src==0)
    ok0 = len(i0)==len(ml) and all(np.array_equal(f[i], np.asarray(mf)[mpos[keys[i]]]) for i in i0)
    i1 = np.flatnonzero(src==1); ok1=True
    if len(i1):
        ff, fl = pickle.load(open(RUN.parents[1]/'features/uni2_all_patches'/f'{s}.pkl','rb'))
        fpos = {tuple(map(int,d['wsi_loc'])):i for i,d in enumerate(fl)}
        ff = np.asarray(ff); ok1 = all(np.array_equal(f[i], ff[fpos[keys[i]]]) for i in i1)
    i2 = np.flatnonzero(src==2); ok2=True
    if len(i2):
        g = np.load(SB/('b3_fallback/out' if s in FB else 'b3/out')/f'{s}.npz')
        gxy, gf = g['xy'], g['feats']
        gpos = {tuple(map(int,p)):i for i,p in enumerate(gxy)}
        ok2 = len(gpos)==len(i2) and all(np.array_equal(f[i], gf[gpos[keys[i]]]) for i in i2)
        gj = json.load(open(SB/('b3_fallback/out' if s in FB else 'b3/out')/f'{s}.json'))
        ok2 = ok2 and 'A100-SXM4-80GB' in json.dumps(gj) and gj.get('gpu_model')=='NVIDIA A100-SXM4-80GB' and gj.get('config_sha256','')[:8]=='8ca3f9b6' and gj.get('env_mismatch') in ([],{},None,'')
    r = dict(slide=s, n=len(f), c=[len(i0),len(i1),len(i2)], sha=ok_sha, dup=ok_dup, n_ok=ok_n, set=ok_set, code=ok_code, cov=ok_cov, b0=ok0, b1=ok1, b2=ok2, s=round(time.time()-t0,1))
    out.append(r); print(r, flush=True)
print('ALL', all(all(v for k,v in r.items() if isinstance(v,bool)) for r in out))
