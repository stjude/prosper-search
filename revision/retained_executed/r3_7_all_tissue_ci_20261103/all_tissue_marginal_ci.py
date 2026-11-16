#!/usr/bin/env python3
"""R3.7 all-tissue arm: marginal whole-patient 95% intervals (2026-11-03).

authorized 2026-11-03 ("proceed with what you recommended on all items";
comment C574 on the Results subtype paragraph: the all-tissue COMET representation is to be the
main-text value, the tumour-filtered one a sensitivity). The all-tissue arm so far had only a
paired-difference interval (r3_7_audit_20261020, INFO block). This script computes the marginal
intervals the main text and Table S26 need, on the EXISTING matrices (no new embeddings, no new
distances). Population/ranking/scoring logic is copied from r3_7_audit_20261020/trace_r3_7.py
(same plurality tie rule, same stable argsort, same same-patient exclusion), run on
  * the all-tissue UNI2-OT matrix (the target), and
  * the tumour-only UNI2-OT matrix (reproduction check against the accepted Table S26 values
    in r3_7_fine_label_20261003/summary_all_arms.csv, seed 123 / B 5000, accepted route).
Bootstrap: whole patients drawn with replacement (multinomial multiplicities as query weights),
PCG64 seed 123, B = 5,000 draws shared by both matrices (so the paired difference uses the same
draws), percentile 2.5 / 97.5. Arms: s61_int_2c (ERMS vs ARMS, COMET) and s64 (HB vs HCC, COMET).
Outputs (this directory): results.json, s26_rows_all_tissue.csv (S26 body_raw column layout),
table.md, run.log, SHA256SUMS.txt.
"""
import hashlib, json, os, pickle, sys, time, platform
import numpy as np
import pandas as pd

ROOT = "/PROSPER_WORKSPACE"
OUT = f"{ROOT}/prosper_revision/prosper_revision_output"
HERE = os.path.dirname(os.path.abspath(__file__))
FINE = f"{OUT}/r3_7_fine_label_20261003"
INV = f"{OUT}/s6_0_label_inventory"
META = f"{ROOT}/comet-feature-gen/comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv"
TUMOR_ONLY = f"{ROOT}/comet-feature-gen/comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine_tumor_only_no_same_patient.pkl"
ALL_TISSUE = f"{ROOT}/comet-feature-gen/comet-uni2/output/similarity_matrix/combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl"
SEED = int(os.environ.get("CI_SEED", 123))
B = int(os.environ.get("CI_B", 5000))
TOL_POINT, TOL_CI, TOL_CI_SMALL, TOL_CI_TINY = 0.0006, 0.006, 0.012, 0.015

t0 = time.time()
log = open(f"{HERE}/run.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()
checks = []
def check(name, exp, obs, tol=0.0):
    ok = abs(float(exp) - float(obs)) <= tol + 1e-12
    checks.append(dict(status="PASS" if ok else "FAIL", check=name, expected=float(exp), observed=float(obs), tol=tol))
    if not ok: say(f"  FAIL {name}: expected {exp} observed {obs}")
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()

say(f"host {platform.node()} python {sys.version.split()[0]} numpy {np.__version__} pandas {pd.__version__} seed {SEED} B {B}")
def load(p):
    M = pickle.load(open(p, "rb")); M.index = M.index.astype(str); M.columns = M.columns.astype(str); return M
tum, allt = load(TUMOR_ONLY), load(ALL_TISSUE)
assert tum.shape == (2921, 2921) and allt.shape == (2921, 2921)
inputs = {k: {"path": p, "sha256": sha(p)} for k, p in [("tumour_only_matrix", TUMOR_ONLY), ("all_tissue_matrix", ALL_TISSUE),
          ("frozen_meta", META), ("accepted_summary", f"{FINE}/summary_all_arms.csv")]}
say(f"matrices loaded ({time.time()-t0:.0f}s)")

ARMS = {"s61_int_2c": ("manifest_s61_rms_comet.csv", ["ERMS", "ARMS"], "ARMS"),
        "s64": ("manifest_s64_hbhcc_comet.csv", ["HB", "HCC"], "HCC")}
def build_population(arm):
    man, labels, pos = ARMS[arm]
    m = pd.read_csv(f"{INV}/{man}", dtype=str)
    inputs[f"manifest_{arm}"] = {"path": f"{INV}/{man}", "sha256": sha(f"{INV}/{man}")}
    m = m[m["sample_type"] != "Xenograft"]
    m = m[m["label"].isin(labels)].copy()
    return m.reset_index(drop=True)

def rank_block(D, qids, rids, qpat, rpat, k=10):
    X = D.loc[qids, rids].to_numpy(dtype=float)
    same = (np.asarray(qpat)[:, None] == np.asarray(rpat)[None, :])
    X = np.where(same, np.inf, X); X = np.where(np.isnan(X), np.inf, X)
    rpos = {r: i for i, r in enumerate(rids)}
    for i, q in enumerate(qids):
        if q in rpos: X[i, rpos[q]] = np.inf
    order = np.argsort(X, axis=1, kind="stable")[:, :k]
    return np.asarray(rids, dtype=object)[order]

def plurality(labs):
    cnt = {}
    for l in labs: cnt[l] = cnt.get(l, 0) + 1
    mx = max(cnt.values())
    for l in labs:
        if cnt[l] == mx: return l

def auroc(score, ispos, w=None):
    score = np.asarray(score, float); ispos = np.asarray(ispos, bool)
    w = np.ones(len(score)) if w is None else np.asarray(w, float)
    sp, wp, sn, wn = score[ispos], w[ispos], score[~ispos], w[~ispos]
    if wp.sum() == 0 or wn.sum() == 0: return np.nan
    gt = (sp[:, None] > sn[None, :]).astype(float) + 0.5 * (sp[:, None] == sn[None, :])
    return float((wp[:, None] * wn[None, :] * gt).sum() / (wp.sum() * wn.sum()))

def score(nb, truth, lab_of, pos):
    L5 = np.vectorize(lab_of.get)(nb[:, :5])
    plur = np.array([plurality(list(r)) for r in L5])
    return dict(L5=L5, plur=plur, hit1=(L5[:, 0] == truth).astype(float), p5=(L5 == truth[:, None]).mean(1),
                plur_ok=(plur == truth).astype(float), pos_share=(L5 == pos).mean(1))

def boot_weights(pats, rng):
    upat, pidx = np.unique(pats, return_inverse=True); n = len(upat)
    return rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)[:, pidx]
def wci(ind, W, mask=None):
    if mask is not None: W = W[:, mask]; ind = ind[mask]
    s = W.sum(1); est = np.where(s > 0, (W @ ind) / np.where(s > 0, s, 1), np.nan)
    return float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5)), est

def summarise(arm, sc, truth, pats, labels, pos, W, rids_n, rpat_n):
    r = {"n_queries": len(truth), "n_patients": len(set(pats)), "n_reference_slides": rids_n, "n_reference_patients": rpat_n,
         "class_counts": {l: int((truth == l).sum()) for l in labels}}
    maj = max(r["class_counts"], key=r["class_counts"].get)
    r["majority_class"], r["majority_share"] = maj, r["class_counts"][maj] / len(truth)
    for key, ind in (("plurality", sc["plur_ok"]), ("hit1", sc["hit1"]), ("p5", sc["p5"])):
        r[key] = float(ind.mean()); lo, hi, _ = wci(ind, W); r[key + "_ci"] = (lo, hi)
    ispos = truth == pos
    r["auroc"] = auroc(sc["pos_share"], ispos)
    a = np.array([auroc(sc["pos_share"], ispos, W[b]) for b in range(B)])
    r["auroc_ci"] = (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))
    per, macro_draws = {}, {"plurality": [], "hit1": [], "p5": []}
    for l in labels:
        mask = truth == l
        e = {"n": int(mask.sum()), "n_patients": len(set(pats[mask]))}
        for key, ind in (("plurality", sc["plur_ok"]), ("hit1", sc["hit1"]), ("p5", sc["p5"])):
            e[key] = float(ind[mask].mean()); lo, hi, est = wci(ind, W, mask); e[key + "_ci"] = (lo, hi); macro_draws[key].append(est)
        e["confusion"] = {m: int(((truth == l) & (sc["plur"] == m)).sum()) for m in labels}
        per[l] = e
    r["per_class"] = per; r["macro"] = {}
    for key in macro_draws:
        r["macro"][key] = float(np.mean([per[l][key] for l in labels]))
        est = np.nanmean(np.vstack(macro_draws[key]), axis=0)
        r["macro"][key + "_ci"] = (float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5)))
    return r

meta = pd.read_csv(META, dtype=str)
summ = pd.read_csv(f"{FINE}/summary_all_arms.csv")
rng = np.random.default_rng(np.random.PCG64(SEED))
results = {"seed": SEED, "B": B, "inputs": inputs, "arms": {}}
rows = []
for arm, (man, labels, pos) in ARMS.items():
    say(f"\n=== {arm} ({' vs '.join(labels)}) ===")
    pop = build_population(arm)
    qids = pop["slide_id"].tolist(); truth = pop["label"].to_numpy()
    lab_of = dict(zip(pop["slide_id"], pop["label"])); patq = dict(zip(pop["slide_id"], pop["patient_id"]))
    qpat = [patq[q] for q in qids]; pats = np.array(qpat)
    rids, rpat = qids, qpat
    W = boot_weights(pats, rng)          # one set of draws, shared by both matrices
    say(f"  queries {len(qids)} patients {len(set(qpat))} reference {len(rids)}  draws {W.shape}")
    out = {}
    for tag, D in (("tumour_only", tum), ("all_tissue", allt)):
        nb = rank_block(D, qids, rids, qpat, rpat)
        sc = score(nb, truth, lab_of, pos)
        r = summarise(arm, sc, truth, pats, labels, pos, W, len(rids), len(set(rpat)))
        out[tag] = r; out[tag + "_sc"] = sc
        say(f"  [{tag}] plurality {r['plurality']:.4f} ({r['plurality_ci'][0]:.4f}, {r['plurality_ci'][1]:.4f})  hit1 {r['hit1']:.4f} ({r['hit1_ci'][0]:.4f}, {r['hit1_ci'][1]:.4f})  "
            f"P@5 {r['p5']:.4f} ({r['p5_ci'][0]:.4f}, {r['p5_ci'][1]:.4f})  AUROC {r['auroc']:.4f} ({r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f})  macro plur {r['macro']['plurality']:.4f}")
        for l in labels:
            e = r["per_class"][l]
            say(f"     {l}: n {e['n']}/{e['n_patients']} plur {e['plurality']:.4f} ({e['plurality_ci'][0]:.4f}, {e['plurality_ci'][1]:.4f}) hit1 {e['hit1']:.4f} P@5 {e['p5']:.4f} conf {e['confusion']}")
    # paired difference all-tissue minus tumour-only (same draws)
    d = out["all_tissue_sc"]["plur_ok"] - out["tumour_only_sc"]["plur_ok"]
    lo, hi, _ = wci(d, W)
    out["delta_plurality_all_tissue_minus_tumour_only"] = {"point": float(d.mean()), "ci": (lo, hi)}
    say(f"  paired delta plurality (all-tissue − tumour-only) {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    # reproduction check: tumour-only vs accepted summary (seed 123 / B 5000, accepted route)
    s = summ[summ.arm == arm].set_index("class"); r = out["tumour_only"]
    tolci = TOL_CI if len(qids) >= 50 else TOL_CI_SMALL
    check(f"{arm} tumour-only plurality point vs accepted", s.loc["ALL", "plurality5_recall"], r["plurality"], TOL_POINT)
    check(f"{arm} tumour-only plurality CI lo vs accepted", s.loc["ALL", "plurality5_recall_lo"], r["plurality_ci"][0], tolci)
    check(f"{arm} tumour-only plurality CI hi vs accepted", s.loc["ALL", "plurality5_recall_hi"], r["plurality_ci"][1], tolci)
    check(f"{arm} tumour-only hit1 point vs accepted", s.loc["ALL", "hit1"], r["hit1"], TOL_POINT)
    check(f"{arm} tumour-only P@5 point vs accepted", s.loc["ALL", "precision5"], r["p5"], TOL_POINT)
    check(f"{arm} tumour-only AUROC point vs accepted", s.loc["ALL", "auroc_top5_pos_share"], r["auroc"], TOL_POINT)
    check(f"{arm} tumour-only AUROC CI lo vs accepted", s.loc["ALL", "auroc_lo"], r["auroc_ci"][0], tolci)
    check(f"{arm} tumour-only AUROC CI hi vs accepted", s.loc["ALL", "auroc_hi"], r["auroc_ci"][1], tolci)
    for l in labels:
        tl = TOL_CI if r["per_class"][l]["n"] >= 50 else TOL_CI_SMALL
        if r["per_class"][l]["n_patients"] < 25: tl = TOL_CI_TINY
        check(f"{arm} tumour-only {l} plurality point vs accepted", s.loc[l, "plurality5_recall"], r["per_class"][l]["plurality"], TOL_POINT)
        check(f"{arm} tumour-only {l} plurality CI lo vs accepted", s.loc[l, "plurality5_recall_lo"], r["per_class"][l]["plurality_ci"][0], tl)
        check(f"{arm} tumour-only {l} plurality CI hi vs accepted", s.loc[l, "plurality5_recall_hi"], r["per_class"][l]["plurality_ci"][1], tl)
    # cross-check all-tissue points against the 10-20 audit INFO values
    aud = json.load(open(f"{OUT}/r3_7_audit_20261020/results_seed20261021_B5000.json"))["results"][arm]["all_tissue_matrix"]
    check(f"{arm} all-tissue plurality point vs 10-20 audit INFO", aud["plurality"], out["all_tissue"]["plurality"], 1e-9)
    check(f"{arm} all-tissue hit1 point vs 10-20 audit INFO", aud["hit1"], out["all_tissue"]["hit1"], 1e-9)
    check(f"{arm} all-tissue AUROC point vs 10-20 audit INFO", aud["auroc"], out["all_tissue"]["auroc"], 1e-9)
    check(f"{arm} paired delta point vs 10-20 audit INFO", aud["delta_plurality"], d.mean(), 1e-9)
    for tag in ("tumour_only", "all_tissue"):
        r = out[tag]
        def row(cls, e, n, npat, nref, nrefp, au=(np.nan, np.nan, np.nan)):
            rows.append(dict(arm=arm, representation=tag, **{"class": cls}, unit="patient", n_queries=n, n_patients=npat,
                             n_reference_slides=nref, n_reference_patients=nrefp,
                             plurality5_recall=e["plurality"], plurality5_recall_lo=e["plurality_ci"][0], plurality5_recall_hi=e["plurality_ci"][1],
                             hit1=e["hit1"], hit1_lo=e["hit1_ci"][0], hit1_hi=e["hit1_ci"][1],
                             precision5=e["p5"], precision5_lo=e["p5_ci"][0], precision5_hi=e["p5_ci"][1],
                             auroc_top5_pos_share=au[0], auroc_lo=au[1], auroc_hi=au[2],
                             confusion=";".join(f"{l} " + "/".join(str(r["per_class"][l]["confusion"][m]) for m in labels) for l in labels) if cls == "ALL" else ""))
        row("ALL", r, r["n_queries"], r["n_patients"], r["n_reference_slides"], r["n_reference_patients"], (r["auroc"], *r["auroc_ci"]))
        row("MACRO", r["macro"], r["n_queries"], np.nan, np.nan, np.nan)
        for l in labels:
            e = r["per_class"][l]; row(l, e, e["n"], e["n_patients"], np.nan, np.nan)
    results["arms"][arm] = {k: v for k, v in out.items() if not k.endswith("_sc")}

results["checks"] = checks
nfail = sum(c["status"] == "FAIL" for c in checks)
say(f"\nchecks PASS {len(checks)-nfail} FAIL {nfail}  ({time.time()-t0:.0f}s)")
json.dump(results, open(f"{HERE}/results.json", "w"), indent=1, default=float)
pd.DataFrame(rows).to_csv(f"{HERE}/s26_rows_all_tissue.csv", index=False)
# formatted table
def f3(x): return f"{x:.3f}"
def ci(e, k): return f"{f3(e[k])} [{f3(e[k+'_ci'][0])}, {f3(e[k+'_ci'][1])}]"
md = ["| Arm | Representation | Row | n slides / patients | Plurality@5 [95% CI] | hit@1 [95% CI] | P@5 [95% CI] | AUROC [95% CI] |", "|---|---|---|---|---|---|---|---|"]
for arm, (man, labels, pos) in ARMS.items():
    for tag in ("all_tissue", "tumour_only"):
        r = results["arms"][arm][tag]
        md.append(f"| {arm} | {tag} | all | {r['n_queries']} / {r['n_patients']} | {ci(r,'plurality')} | {ci(r,'hit1')} | {ci(r,'p5')} | {f3(r['auroc'])} [{f3(r['auroc_ci'][0])}, {f3(r['auroc_ci'][1])}] |")
        md.append(f"| | | macro | — | {ci(r['macro'],'plurality')} | {ci(r['macro'],'hit1')} | {ci(r['macro'],'p5')} | — |")
        for l in labels:
            e = r["per_class"][l]; md.append(f"| | | {l} | {e['n']} / {e['n_patients']} | {ci(e,'plurality')} | {ci(e,'hit1')} | {ci(e,'p5')} | — |")
    dd = results["arms"][arm]["delta_plurality_all_tissue_minus_tumour_only"]
    md.append(f"| {arm} | paired Δ plurality (all-tissue − tumour-only) | | | {dd['point']:+.3f} [{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}] | | | |")
open(f"{HERE}/table.md", "w").write("\n".join(md) + "\n")
say("\n".join(md))
log.close()
