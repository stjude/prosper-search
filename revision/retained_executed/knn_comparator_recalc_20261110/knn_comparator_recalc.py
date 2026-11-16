#!/usr/bin/env python3
"""E1-KNN: reproducible plurality kNN comparator for Table S26 from released S27/S29 labels,
Bayesian argmax from S28/S30, G1/G2 counts and Mann-Whitney on LR1t2. Read-only inputs."""
import json, hashlib, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
P = sys.argv[1]; OUT = sys.argv[2]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
files = {n: f"{P}/Supplementary_Table_{n}.xlsx" for n in ["S26","S27","S28","S29","S30"]}
lab = {5: pd.read_excel(files["S27"], sheet_name="Sheet1"), 18: pd.read_excel(files["S29"], sheet_name="Sheet1")}
sc  = {5: pd.read_excel(files["S28"], sheet_name="Sheet1"), 18: pd.read_excel(files["S30"], sheet_name="Sheet1")}
s26 = pd.read_excel(files["S26"], sheet_name="model_comparison_2766").set_index("N")
groups = sorted(lab[5]["disease"].unique())
def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p); f1s=[]; recs=[]
    for g in groups:
        tp=np.sum((p==g)&(y==g)); fp=np.sum((p==g)&(y!=g)); fn=np.sum((p!=g)&(y==g))
        f1s.append(0.0 if tp==0 else 2*tp/(2*tp+fp+fn)); recs.append(tp/(tp+fn))
    return dict(accuracy=float(np.mean(y==p)), macro_f1=float(np.mean(f1s)), balanced_accuracy=float(np.mean(recs)))
def plurality_closest(labels):
    c = Counter(labels); m = max(c.values()); tied = [l for l in c if c[l]==m]
    if len(tied)==1: return tied[0], False
    for l in labels:
        if l in tied: return l, True
def plurality_strict(labels):
    c = Counter(labels); m = max(c.values()); tied=[l for l in c if c[l]==m]
    return (tied[0] if len(tied)==1 else "__TIE__"), len(tied)>1
res = {"inputs": {n: sha(f) for n,f in files.items()}, "groups": groups, "n_rows": int(len(lab[5])), "k": {}}
rows=[]
for k in (5,18):
    L = lab[k]; S = sc[k]
    assert list(L["slides"])==list(S["slides"]), "slide order differs"
    y = L["disease"].values
    ncols = [f"N{i}" for i in range(k)]
    knn_c, tie_c = zip(*[plurality_closest(list(r)) for r in L[ncols].values])
    knn_s, _     = zip(*[plurality_strict(list(r)) for r in L[ncols].values])
    scols = [c for c in S.columns if c not in ("Unnamed: 0","slides")]
    bay = S[scols].idxmax(axis=1).values
    out = {"plurality_closest_rank": metrics(y, knn_c), "plurality_ties_as_wrong": metrics(y, knn_s),
           "bayesian_argmax": metrics(y, bay), "n_tied_queries": int(sum(tie_c)),
           "historical_S26": {"macro_f1": float(s26.loc["F1-no tie break", k]), "accuracy": float(s26.loc["accuracy-no tie break", k]),
                               "balanced_accuracy": float(s26.loc["balanced accuracy-no tie break", k])},
           "historical_S26_bayesian": {"macro_f1": float(s26.loc["F1-Bayesian", k]), "accuracy": float(s26.loc["accuracy-Bayesian", k]),
                               "balanced_accuracy": float(s26.loc["balanced accuracy-Bayesian", k])}}
    # fidelity: Bayesian rows
    fid = all(abs(out["bayesian_argmax"][m]-out["historical_S26_bayesian"][m])<5e-4 for m in ("macro_f1","accuracy","balanced_accuracy"))
    kc = np.asarray(knn_c); 
    g1 = (bay==y)&(kc!=y); g2 = (kc==y)&(bay!=y)
    saved_g1 = (L["candidate"].astype(str).str.lower()=="yes").values; saved_g2=(L["reverse-candidate"].astype(str).str.lower()=="yes").values
    lr = L["LR1t2"].astype(float).values
    out.update({"G1_reproducible": int(g1.sum()), "G2_reproducible": int(g2.sum()),
                "G1_saved_flags": int(saved_g1.sum()), "G2_saved_flags": int(saved_g2.sum()),
                "flag_agreement_G1": int((g1&saved_g1).sum()), "flag_agreement_G2": int((g2&saved_g2).sum()),
                "mannwhitney_p_reproducible": float(mannwhitneyu(lr[g1], lr[g2], alternative="two-sided").pvalue),
                "mannwhitney_p_saved_flags": float(mannwhitneyu(lr[saved_g1], lr[saved_g2], alternative="two-sided").pvalue),
                "LR1t2_median_G1_G2_reproducible": [float(np.median(lr[g1])), float(np.median(lr[g2]))],
                "fidelity_bayesian_rows": bool(fid), "fidelity_saved_flags": bool(saved_g1.sum() in (103,125) and saved_g2.sum() in (99,47))})
    res["k"][str(k)] = out
    for name in ("plurality_closest_rank","plurality_ties_as_wrong","bayesian_argmax","historical_S26","historical_S26_bayesian"):
        rows.append(dict(k=k, rule=name, **out[name]))
res["fidelity_pass"] = all(res["k"][k]["fidelity_bayesian_rows"] and res["k"][k]["fidelity_saved_flags"] for k in res["k"])
json.dump(res, open(f"{OUT}/results.json","w"), indent=1); pd.DataFrame(rows).to_csv(f"{OUT}/results.csv", index=False)
print(json.dumps(res, indent=1))
