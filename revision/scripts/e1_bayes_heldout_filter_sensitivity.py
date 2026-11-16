#!/usr/bin/env python3
"""E1-BAYHO: Bayesian tumor-group probabilities under the patient-held-out filtering masks.

Bounded sensitivity readback (record: [internal record], section "E1-BAYHO", pre-registered 2026-11-14 11:15 CST;
authorized 2026-11-14). Supplementary Tables S26-S30 and the case-study probabilities of
Results Units 9-10 / Fig. 7 use the submitted tumor-only matrix, whose tile classifier was not held out by
patient. This script recomputes the same quantities from the retained patient-held-out matrix of R2.1-TF2
(UNI2, Optimal Transport cosine, both-sided, 100% of labels) on the common 2,604-slide, 14-group tumor
population and reports how far they move. It changes no deliverable.

Configurations (UNI2, OT cosine, same-patient excluded, T = 20; the k nearest references are weighted and the
(k + 1)st sets the distance scale):
  S  submitted tumor-only matrix as published (2,766 queries, 15 groups) - fidelity only
  A  submitted tumor-only matrix restricted to the 2,604-slide tumor population
  B  patient-held-out matrix (R2.1-TF2) on that population
  C  all-tissue matrix restricted to that population (context)

Read-only on every input; refuses to overwrite an existing run directory. Login node, one process.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import hashlib
import json
import pickle
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAORAN = "/PROSPER_WORKSPACE"
T = 20.0
SPAN_EPS = 1e-10
SENTINEL = 1e5
KS = (5, 18)
K_SWEEP = tuple(range(2, 26))
KMAX = max(K_SWEEP)
BOOT_B = 5000
BOOT_SEED = 123
LR_EPS = 0.001

CASES = [  # (identifier, scan id, recorded group)
    ("SJOS001106_D1", "15512", "OS"),
    ("SJDSRCT073174_D1", "15390", "DSRCT"),
    ("SJMPNST076015_D1", "12628", "MPNST"),
    ("SJHB075481_D1", "9937", "MRT"),
    ("SJMRT031055_D1", "10033", "MRT"),
    ("SJWLM056159_D2", "13093", "WT"),
    ("SJMPNST073103_R1", "16137", "MPNST"),
]


def inputs(out_root):
    tf2 = os.path.join(out_root, "r2_1_filter_tf2_20261016")
    pkg = os.path.join(out_root, "supp_package_20261110", "supplementary_tables")
    sim = os.path.join(HAORAN, "comet-feature-gen", "comet-uni2", "output", "similarity_matrix")
    return {
        "matrix_submitted_tumor_only": (
            os.path.join(sim, "combined_similarity_matrix_uni2_ot_cosine_tumor_only_no_same_patient.pkl"),
            "03e0e2c7205fde649295b98febf822218814ffb7f487a2ce71f71017a5b058cc"),
        "matrix_all_tissue": (
            os.path.join(sim, "combined_similarity_matrix_uni2_ot_cosine_no_same_patient.pkl"),
            "a288d8926c8715c86ae3063ff68d9f7b0c6b365eaa77549f45aa02142d2e1714"),
        "matrix_heldout_both_sided_b100": (
            os.path.join(tf2, "s5", "combined_001", "uni2", "b100", "both_sided_ot",
                         "combined_similarity_matrix_tf2_uni2_b100_both_sided_ot_cosine_no_same_patient.pkl"),
            "9f2d2449a22a037174c23f8344eca1d995da3ef61d6b334304ff4b2c9d641ef8"),
        "population": (
            os.path.join(out_root, "r2_1_cpu_20261002", "filter_attempt_001", "retrieval_population.csv"),
            "c7cae1f0d855952e84e85aa90c7ef7d8a622086273c113285655c96be946d97b"),
        "query_assignment": (
            os.path.join(tf2, "s0", "run_002", "query_assignment.csv"),
            "8fb84958c2580a90392598aa91ff728834b2fbcd5d9aaae6f3b6610f8d5bc906"),
        "ranked_lists": (
            os.path.join(tf2, "s6", "evaluation_primary_001", "ranked_lists.csv.gz"),
            "db48217df9debfabee43d6c224e5003a5a3a8a4313fe8624f617111a46f8952a"),
        "library_map": (
            os.path.join(HAORAN, "comet-feature-gen", "comet-analysis", "data",
                         "final_scan_id_disease_map_with_sample_id_patient_id.csv"),
            "45f1466c158f41ceea11abe06138356597a65ce1e97d2b072b780951864582b9"),
        "S26": (os.path.join(pkg, "Supplementary_Table_S26.xlsx"),
                "497fd03954ee481b92e200d30dbeb148d7deb22a7b162b4a3e22abe2f6fb8cb4"),
        "S28": (os.path.join(pkg, "Supplementary_Table_S28.xlsx"),
                "d09d256aa35efa2d5035ec138cb8b016172e8662cd239aeef15b49bf2a83b9a3"),
        "S30": (os.path.join(pkg, "Supplementary_Table_S30.xlsx"),
                "8f25fa3dd1aa5076725e74ef3c5723c97011cdf1ce91cb73a14126955ccb7eaa"),
        "c10_tumor_only": (
            os.path.join(out_root, "c10_calibration", "calibration_uni2_ot_cosine_tumor_only.json"),
            "babda45098ff6ea1995759763a7fe5e5c5e34bfe915af50e7eb93244127d07c3"),
        "case_metadata": (
            os.path.join(REPO, "data", "ALIdf3Ali.csv"),
            "dff01f16591d51e76ff197bf6b1d3fb64989f9a8f3264ec6acfa727ea05750ae"),
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- core computations

def kernel_weights(d, k, temperature=T):
    """Weights of the k nearest references. d: ascending distances of the k + 1 nearest references.

    w_i = exp(-tau (d_i - d_1)), tau = ln(T) / (d_{k+1} - d_1): the nearest reference has weight 1 and a
    reference at the (k + 1)st distance would have weight 1/T. A span below SPAN_EPS gives equal weights.
    """
    d = np.asarray(d, dtype=float)
    span = d[k] - d[0]
    if span < SPAN_EPS:
        return np.ones(k)
    return np.exp(-np.log(temperature) / span * (d[:k] - d[0]))


def prepare(matrix, ids, labels, patients, classes, kmax=KMAX):
    """Sorted neighbor table for one configuration.

    matrix: square DataFrame of distances (index/columns str slide ids); ids: query = reference order;
    labels, patients: aligned with ids. Non-finite cells, cells >= SENTINEL and same-patient cells are excluded.
    Returns dict with dist (n, kmax + 1; inf-padded), lab (class index; -1 padded), nbr (position in ids; -1),
    n_finite per query and the number of cells masked only by the patient rule.
    """
    ids = [str(s) for s in ids]
    x = matrix.loc[ids, ids].to_numpy(dtype=float, copy=True)
    bad = ~np.isfinite(x) | (x >= SENTINEL)
    pat = np.asarray(patients)
    same = pat[:, None] == pat[None, :]
    extra = int((same & ~bad).sum())
    x[bad | same] = np.inf
    ci = {c: i for i, c in enumerate(classes)}
    lab_idx = np.array([ci[l] for l in labels])
    n, m = len(ids), kmax + 1
    order = np.argsort(x, axis=1, kind="stable")[:, :m]
    dist = np.take_along_axis(x, order, axis=1)
    fin = np.isfinite(dist)
    lab = np.where(fin, lab_idx[order], -1)
    nbr = np.where(fin, order, -1)
    return {"ids": ids, "y": lab_idx, "patients": pat, "classes": list(classes), "dist": dist, "lab": lab,
            "nbr": nbr, "n_finite": fin.sum(1), "cells_masked_by_patient_rule_only": extra}


def bayes(conf, k):
    """Probabilities (n, C) from the k nearest references; rows without k + 1 finite neighbors are NaN."""
    n, c = len(conf["ids"]), len(conf["classes"])
    p = np.full((n, c), np.nan)
    ok = conf["n_finite"] >= k + 1
    for i in np.flatnonzero(ok):
        w = kernel_weights(conf["dist"][i], k)
        s = np.bincount(conf["lab"][i, :k], weights=w, minlength=c)
        p[i] = s / s.sum()
    return p, ok


def argmax_pred(p, ok):
    """Predicted class index; exact probability ties go to the first class in (alphabetical) class order."""
    pred = np.full(len(p), -1)
    pred[ok] = np.argmax(p[ok], axis=1)
    return pred


def plurality_closest(lab_row):
    """Plurality vote over neighbor class indices in rank order; count ties go to the closest-ranked tied label."""
    cnt = collections.Counter(lab_row.tolist())
    top = max(cnt.values())
    for l in lab_row.tolist():
        if cnt[l] == top:
            return l
    raise AssertionError("unreachable")


def plurality_pred(conf, k):
    ok = conf["n_finite"] >= k + 1
    pred = np.full(len(conf["ids"]), -1)
    for i in np.flatnonzero(ok):
        pred[i] = plurality_closest(conf["lab"][i, :k])
    return pred, ok


def metrics(y, pred, n_classes, w=None):
    """Accuracy, macro F1 (classes present in truth or prediction) and balanced accuracy (classes with support).

    Same definitions as sklearn accuracy_score, f1_score(average='macro') and balanced_accuracy_score; w are
    optional non-negative query weights (bootstrap multiplicities).
    """
    cm = np.bincount(np.asarray(y) * n_classes + np.asarray(pred), weights=w,
                     minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(float)
    tp, sup, pp = np.diag(cm), cm.sum(1), cm.sum(0)
    present = (sup + pp) > 0
    f1 = np.zeros(n_classes)
    f1[present] = 2 * tp[present] / (sup[present] + pp[present])
    has = sup > 0
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1[present].mean()),
            "balanced_accuracy": float((tp[has] / sup[has]).mean())}


def per_class_f1(y, pred, n_classes):
    cm = np.bincount(np.asarray(y) * n_classes + np.asarray(pred), minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes).astype(float)
    tp, sup, pp = np.diag(cm), cm.sum(1), cm.sum(0)
    return np.where((sup + pp) > 0, 2 * tp / np.maximum(sup + pp, 1e-300), np.nan), sup


def top_two_ratio(p):
    s = -np.sort(-p, axis=1)
    return (s[:, 0] + LR_EPS) / (s[:, 1] + LR_EPS)


def g_counts(y, bayes_pred, knn_pred, p):
    """Corrected (Bayesian right, vote wrong) and introduced (vote right, Bayesian wrong) with the ratio test."""
    g1 = (bayes_pred == y) & (knn_pred != y)
    g2 = (bayes_pred != y) & (knn_pred == y)
    lr = top_two_ratio(p)
    pval = float(mannwhitneyu(lr[g1], lr[g2]).pvalue) if g1.any() and g2.any() else float("nan")
    return {"corrected": int(g1.sum()), "introduced": int(g2.sum()), "mann_whitney_p": pval,
            "median_ratio_corrected": float(np.median(lr[g1])) if g1.any() else float("nan"),
            "median_ratio_introduced": float(np.median(lr[g2])) if g2.any() else float("nan")}


def patient_bootstrap(y, preds, patients, n_classes, b=BOOT_B, seed=BOOT_SEED):
    """Whole-patient percentile bootstrap with fixed predictions.

    preds: dict name -> prediction array (all on the same queries). Returns dict name -> (b, 3) array of
    accuracy, macro F1, balanced accuracy; patients are drawn with replacement and keep all their queries.
    """
    uniq, inv = np.unique(np.asarray(patients), return_inverse=True)
    rng = np.random.default_rng(seed)
    out = {name: np.empty((b, 3)) for name in preds}
    for r in range(b):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float)
        for name, pr in preds.items():
            m = metrics(y, pr, n_classes, w)
            out[name][r] = (m["accuracy"], m["macro_f1"], m["balanced_accuracy"])
    return out


def case_reading(name, k5, k18):
    """Pre-registered reading checks. k5, k18: dicts with 'probs' (group -> p, descending) and 'counts'."""
    def top(d):
        return list(d["probs"].keys())

    def most_frequent(d):
        c = d["counts"]
        m = max(c.values())
        return sorted(g for g, v in c.items() if v == m)

    if name == "SJOS001106_D1":
        return {"OS top at k=5": top(k5)[0] == "OS"}
    if name == "SJDSRCT073174_D1":
        return {"DSRCT top at k=5": top(k5)[0] == "DSRCT", "RMS most frequent label at k=5": most_frequent(k5) == ["RMS"]}
    if name == "SJMPNST076015_D1":
        pr = list(k5["probs"].values())
        gap = pr[0] - (pr[1] if len(pr) > 1 else 0.0)
        return {"MPNST and RMS top two at k=5": set(top(k5)[:2]) == {"MPNST", "RMS"}, "top-two gap <= 0.10 at k=5": gap <= 0.10}
    if name == "SJHB075481_D1":
        return {"LIVER top at k=18": top(k18)[0] == "LIVER", "MRT in top three at k=18": "MRT" in top(k18)[:3]}
    if name == "SJMRT031055_D1":
        return {"MRT top at k=18": top(k18)[0] == "MRT"}
    if name == "SJWLM056159_D2":
        return {"RMS top at k=18": top(k18)[0] == "RMS", "WT in top three at k=18": "WT" in top(k18)[:3]}
    if name == "SJMPNST073103_R1":
        return {"RMS top at k=18": top(k18)[0] == "RMS", "MPNST in top three at k=18": "MPNST" in top(k18)[:3]}
    raise KeyError(name)


def case_row(conf, p, ok, sid, k):
    if sid not in conf["ids"]:
        return None
    i = conf["ids"].index(sid)
    if not ok[i]:
        return None
    cls = conf["classes"]
    order = np.argsort(-p[i], kind="stable")
    probs = {cls[j]: float(p[i, j]) for j in order if p[i, j] > 0}
    counts = dict(collections.Counter(cls[j] for j in conf["lab"][i, :k]))
    return {"probs": probs, "counts": counts, "nearest_label": cls[conf["lab"][i, 0]]}


# ---------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run_001")
    ap.add_argument("--skip-tests", action="store_true", help="do not run the unit tests (gate F5 then fails)")
    args = ap.parse_args()
    out_root = os.environ.get("PROSPER_OUT", os.path.join(REPO, "prosper_revision_output"))
    dest = os.path.join(out_root, "e1_bayes_heldout_filter_sensitivity_20261114", args.run)
    if os.path.exists(dest):
        sys.exit(f"refusing to overwrite {dest}")
    t0 = time.time()
    gates, res = {}, {}

    # F4 input hashes
    inp = inputs(out_root)
    bind = {}
    for name, (path, expected) in inp.items():
        got = sha256(path)
        bind[name] = {"path": path, "sha256": got, "expected": expected, "match": got == expected}
    gates["F4_input_hashes"] = {"pass": all(v["match"] for v in bind.values()),
                                "mismatches": [k for k, v in bind.items() if not v["match"]]}

    # F5 unit tests
    test_path = os.path.join(REPO, "tests", "test_e1_bayes_heldout_filter_sensitivity.py")
    if args.skip_tests:
        gates["F5_unit_tests"] = {"pass": False, "detail": "skipped"}
        test_log = ""
    else:
        pr = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test_path],
                            capture_output=True, text=True, cwd=REPO,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        test_log = pr.stdout + pr.stderr
        gates["F5_unit_tests"] = {"pass": pr.returncode == 0, "summary": test_log.strip().splitlines()[-1] if test_log.strip() else ""}

    load = lambda key: pickle.load(open(inp[key][0], "rb"))
    m_sub, m_all, m_ho = load("matrix_submitted_tumor_only"), load("matrix_all_tissue"), load("matrix_heldout_both_sided_b100")
    for m in (m_sub, m_all, m_ho):
        m.index = m.index.astype(str)
        m.columns = m.columns.astype(str)
    lib = pd.read_csv(inp["library_map"][0])
    lib["sid"] = lib["Slide.Scan.File"].astype(str)
    g_map, p_map = dict(zip(lib.sid, lib["Tumor.group"])), dict(zip(lib.sid, lib["UID.Subject"]))
    pop = pd.read_csv(inp["population"][0])
    pop["sid"] = pop.slide_id.astype(str)
    qa = pd.read_csv(inp["query_assignment"][0])
    qa["sid"] = qa.slide_id.astype(str)
    annotated = dict(zip(qa.sid, qa.patient_annotated.astype(int)))

    # F7 population labels/patients equal the library map
    lab_ok = int(sum(g_map.get(s) == g for s, g in zip(pop.sid, pop.diagnostic_group)))
    pat_ok = int(sum(p_map.get(s) == p for s, p in zip(pop.sid, pop.patient_id)))
    gates["F7_population_equals_library_map"] = {
        "pass": lab_ok == len(pop) == 2604 and pat_ok == len(pop) and set(qa.sid) == set(pop.sid),
        "n": len(pop), "labels_equal": lab_ok, "patients_equal": pat_ok,
        "groups": int(pop.diagnostic_group.nunique()), "patients": int(pop.patient_id.nunique())}

    # configurations
    cls_s = sorted(set(g_map.values()) - {"MISC"})
    ids_s = [s for s in m_sub.index if g_map.get(s, "MISC") != "MISC"]
    conf_s = prepare(m_sub, ids_s, [g_map[s] for s in ids_s], [p_map[s] for s in ids_s], cls_s)
    cls_t = sorted(pop.diagnostic_group.unique())
    ids_t = pop.sid.tolist()
    lab_t, pat_t = pop.diagnostic_group.tolist(), pop.patient_id.tolist()
    conf = {"A": prepare(m_sub, ids_t, lab_t, pat_t, cls_t), "B": prepare(m_ho, ids_t, lab_t, pat_t, cls_t),
            "C": prepare(m_all, ids_t, lab_t, pat_t, cls_t)}
    res["cells_masked_by_patient_rule_only"] = {"S": conf_s["cells_masked_by_patient_rule_only"],
                                                **{k: v["cells_masked_by_patient_rule_only"] for k, v in conf.items()}}

    # F1 submitted configuration reproduces the packaged tables
    f1 = {}
    ps = {k: bayes(conf_s, k) for k in K_SWEEP}
    ys = conf_s["y"]
    for k, key in ((5, "S28"), (18, "S30")):
        t = pd.read_excel(inp[key][0], sheet_name="Sheet1", index_col=0)
        t.index = t.slides.str.replace(".svs", "", regex=False)
        p, ok = ps[k]
        mine = pd.DataFrame(p[ok], index=np.array(conf_s["ids"])[ok], columns=cls_s)
        same_rows = set(mine.index) == set(t.index)
        diff = float(np.abs(mine.loc[t.index, cls_s].to_numpy() - t[cls_s].to_numpy(dtype=float)).max()) if same_rows else float("inf")
        f1[f"{key}_k{k}"] = {"rows": int(ok.sum()), "same_rows": same_rows, "max_abs_diff": diff}
    s26 = pd.read_excel(inp["S26"][0], sheet_name="model_comparison_2766", header=None)
    ncol = {int(v): j for j, v in s26.iloc[0, 1:].items() if pd.notna(v)}
    row = {str(s26.iloc[i, 0]): i for i in range(1, 7)}
    # Gate (as pre-registered): k = 5 and 18. The other k are a diagnostic only (run_001 stopped because this
    # check had been coded for k = 2-25; the k = 2 column of the submitted sheet is not reproduced, k = 3-25 are).
    d26, d26_by_k = 0.0, {}
    for k in K_SWEEP:
        p, ok = ps[k]
        mk = metrics(ys[ok], argmax_pred(p, ok)[ok], len(cls_s))
        dk = 0.0
        for name, key in (("F1-Bayesian", "macro_f1"), ("accuracy-Bayesian", "accuracy"), ("balanced accuracy-Bayesian", "balanced_accuracy")):
            dk = max(dk, abs(mk[key] - float(s26.iloc[row[name], ncol[k]])))
        d26_by_k[k] = dk
        if k in KS:
            d26 = max(d26, dk)
    f1["S26_bayesian_rows_k5_k18_max_abs_diff"] = d26
    f1["diagnostic_S26_bayesian_rows_max_abs_diff_by_k"] = {str(k): v for k, v in d26_by_k.items()}
    f1["diagnostic_S26_bayesian_k_not_reproduced_5e-7"] = [k for k, v in d26_by_k.items() if v > 5e-7]
    plu = pd.read_excel(inp["S26"][0], sheet_name="plurality_recomputed", header=1)
    dpl, gc = 0.0, {}
    for k in KS:
        p, ok = ps[k]
        kp, _ = plurality_pred(conf_s, k)
        mk = metrics(ys[ok], kp[ok], len(cls_s))
        r = plu[(plu.k == k) & (plu.rule == "plurality_closest_rank")].iloc[0]
        dpl = max(dpl, abs(mk["accuracy"] - r.accuracy), abs(mk["macro_f1"] - r.macro_f1), abs(mk["balanced_accuracy"] - r.balanced_accuracy))
        gc[k] = g_counts(ys[ok], argmax_pred(p, ok)[ok], kp[ok], p[ok])
    f1["S26_plurality_recomputed_max_abs_diff"] = dpl
    f1["corrected_introduced"] = {str(k): [gc[k]["corrected"], gc[k]["introduced"]] for k in KS}
    f1["mann_whitney_p"] = {str(k): gc[k]["mann_whitney_p"] for k in KS}
    f1["pass"] = bool(all(f1[f"{key}_k{k}"]["same_rows"] and f1[f"{key}_k{k}"]["rows"] == 2766 and f1[f"{key}_k{k}"]["max_abs_diff"] <= 1e-9
                          for k, key in ((5, "S28"), (18, "S30")))
                      and d26 <= 5e-7 and dpl <= 5e-7
                      and f1["corrected_introduced"] == {"5": [72, 89], "18": [109, 46]}
                      and abs(gc[5]["mann_whitney_p"] - 0.01977118149) <= 1e-6 and abs(gc[18]["mann_whitney_p"] - 0.001065924675) <= 1e-6)
    gates["F1_submitted_tables_reproduced"] = f1

    # F2 restricted submitted matrix reproduces the C10 tumor-only accuracies (non-returned query incorrect)
    c10 = json.load(open(inp["c10_tumor_only"][0]))
    f2 = {"n_queries_c10": c10["config"]["n_queries"]}
    okf2 = c10["config"]["n_queries"] == len(ids_t)
    for k in KS:
        p, ok = bayes(conf["A"], k)
        acc = float((argmax_pred(p, ok)[ok] == conf["A"]["y"][ok]).sum() / len(ids_t))
        ref = c10["by_N"][str(k)]["accuracy"]
        f2[f"k{k}"] = {"mine": acc, "c10": ref, "returned": int(ok.sum())}
        okf2 = okf2 and abs(acc - ref) <= 1e-6
    f2["pass"] = bool(okf2)
    gates["F2_c10_tumor_only_accuracy"] = f2

    # F3 held-out matrix: ten nearest neighbors equal the retained ranked lists
    rl = pd.read_csv(inp["ranked_lists"][0], dtype={"query": str, "neighbour": str})
    rl = rl[(rl.encoder == "uni2") & (rl.arm == "both_filtered") & (rl.method == "ot_cosine") & (rl.budget == 100)]
    rl = rl.sort_values(["query", "rank"])
    cb = conf["B"]
    pos = {s: i for i, s in enumerate(cb["ids"])}
    ids_arr = np.array(cb["ids"])
    n_rows = len(rl)
    max_d, id_mismatch, tie_perm, grp_mismatch = 0.0, 0, 0, 0
    for q, grp in rl.groupby("query", sort=False):
        i = pos[q]
        r = grp["rank"].to_numpy() - 1
        mine_d = cb["dist"][i, r]
        mine_n = ids_arr[cb["nbr"][i, r]]
        max_d = max(max_d, float(np.abs(mine_d - grp.distance.to_numpy()).max()))
        neq = mine_n != grp.neighbour.to_numpy()
        if neq.any():
            if sorted(mine_n.tolist()) == sorted(grp.neighbour.tolist()) and np.abs(mine_d - grp.distance.to_numpy()).max() <= 1e-12:
                tie_perm += int(neq.sum())
            else:
                id_mismatch += int(neq.sum())
        grp_mismatch += int((np.array([cls_t[j] for j in cb["lab"][i, r]]) != grp.neighbour_group.to_numpy()).sum()) if not neq.any() else 0
    gates["F3_heldout_ranked_lists"] = {
        "pass": bool(n_rows == 26040 and max_d <= 1e-12 and id_mismatch == 0 and grp_mismatch == 0),
        "rows": int(n_rows), "max_abs_distance_diff": max_d, "neighbor_id_mismatches": id_mismatch,
        "rows_permuted_within_exact_ties": tie_perm, "neighbor_group_mismatches": grp_mismatch}

    # common query set and predictions
    kneed = KMAX + 1
    returned = {c: conf[c]["n_finite"] >= kneed for c in conf}
    common = returned["A"] & returned["B"] & returned["C"]
    y = conf["A"]["y"]
    pats = np.array(pat_t)
    ann = np.array([annotated[s] for s in ids_t])
    nc = len(cls_t)
    P = {c: {k: bayes(conf[c], k)[0] for k in K_SWEEP} for c in ("A", "B")}
    P["C"] = {k: bayes(conf["C"], k)[0] for k in KS}
    pred = {c: {k: argmax_pred(P[c][k], common) for k in P[c]} for c in P}
    vote = {c: {k: plurality_pred(conf[c], k)[0] for k in KS} for c in conf}

    # F6 bootstrap determinism (and the bootstrap itself)
    boot_in = {f"{c}_k{k}": pred[c][k][common] for c in ("A", "B") for k in KS}
    boot = patient_bootstrap(y[common], boot_in, pats[common], nc)
    again = patient_bootstrap(y[common], boot_in, pats[common], nc, b=200)
    gates["F6_bootstrap_determinism"] = {"pass": bool(all(np.array_equal(boot[n][:200], again[n]) for n in boot))}

    all_pass = all(g["pass"] for g in gates.values())
    os.makedirs(dest)
    json.dump(bind, open(os.path.join(dest, "input_bindings.json"), "w"), indent=1)
    if test_log:
        open(os.path.join(dest, "pytest.log"), "w").write(test_log)
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": __import__("scipy").__version__, "host": platform.node(), "argv": sys.argv,
           "script_sha256": sha256(os.path.abspath(__file__)), "tests_sha256": sha256(test_path) if os.path.exists(test_path) else None,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0))}
    if not all_pass:
        json.dump({"status": "GATE_FAIL", "gates": gates, "env": env}, open(os.path.join(dest, "results.json"), "w"), indent=1)
        print(json.dumps(gates, indent=1))
        sys.exit(2)

    # E1 metrics and B - A with whole-patient intervals
    names = ("accuracy", "macro_f1", "balanced_accuracy")
    rows = []
    for c in ("A", "B", "C"):
        for k in KS:
            m = metrics(y[common], pred[c][k][common], nc)
            rows.append({"config": c, "k": k, "rule": "bayesian_argmax", "n_queries": int(common.sum()), **m})
            mv = metrics(y[common], vote[c][k][common], nc)
            rows.append({"config": c, "k": k, "rule": "plurality_closest_rank", "n_queries": int(common.sum()), **mv})
    mt = pd.DataFrame(rows)
    e1 = {}
    for k in KS:
        a = mt[(mt.config == "A") & (mt.k == k) & (mt.rule == "bayesian_argmax")].iloc[0]
        b = mt[(mt.config == "B") & (mt.k == k) & (mt.rule == "bayesian_argmax")].iloc[0]
        d = boot[f"B_k{k}"] - boot[f"A_k{k}"]
        e1[str(k)] = {n: {"A": float(a[n]), "B": float(b[n]), "B_minus_A": float(b[n] - a[n]),
                          "ci95": [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))]}
                      for j, n in enumerate(names)}
    res["E1_metrics"] = e1

    # E2 macro F1 by k
    sweep = []
    for c in ("A", "B"):
        for k in K_SWEEP:
            sweep.append({"config": c, "k": k, **metrics(y[common], pred[c][k][common], nc)})
    sw = pd.DataFrame(sweep)
    e2 = {}
    for c in ("A", "B"):
        s = sw[sw.config == c].set_index("k").macro_f1
        e2[c] = {"k_max_f1": int(s.idxmax()), "max_f1": float(s.max()), "f1_k18": float(s.loc[18]),
                 "f1_k18_minus_max": float(s.loc[18] - s.max()), "f1_k5": float(s.loc[5]),
                 "f1_min": float(s.min()), "k_min_f1": int(s.idxmin())}
    res["E2_f1_by_k"] = e2

    # E3 Bayesian versus plurality vote
    e3 = {}
    for c in ("A", "B"):
        for k in KS:
            mb = mt[(mt.config == c) & (mt.k == k) & (mt.rule == "bayesian_argmax")].iloc[0]
            mv = mt[(mt.config == c) & (mt.k == k) & (mt.rule == "plurality_closest_rank")].iloc[0]
            e3[f"{c}_k{k}"] = {"bayesian_minus_plurality": {n: float(mb[n] - mv[n]) for n in names},
                               **g_counts(y[common], pred[c][k][common], vote[c][k][common], P[c][k][common])}
    res["E3_bayesian_vs_plurality"] = e3

    # E4 per-query A versus B, E5 nearest-neighbor equivalence
    e4, e5 = {}, {}
    pq = pd.DataFrame({"query": ids_t, "group": lab_t, "patient": pat_t, "patient_annotated": ann,
                       "in_common_set": common.astype(int)})
    for k in KS:
        same_set = np.array([set(conf["A"]["nbr"][i, :k]) == set(conf["B"]["nbr"][i, :k]) for i in range(len(ids_t))])
        overlap = np.array([len(set(conf["A"]["nbr"][i, :k]) & set(conf["B"]["nbr"][i, :k])) for i in range(len(ids_t))])
        dtop = np.abs(np.nanmax(P["B"][k], axis=1) - np.nanmax(P["A"][k], axis=1))
        tv = 0.5 * np.abs(P["B"][k] - P["A"][k]).sum(1)
        for c in ("A", "B", "C"):
            pq[f"pred_{c}_k{k}"] = [cls_t[j] if j >= 0 else "" for j in pred[c][k]]
            pq[f"top_prob_{c}_k{k}"] = np.nanmax(P[c][k], axis=1)
        for c in ("A", "B"):
            pq[f"vote_{c}_k{k}"] = [cls_t[j] if j >= 0 else "" for j in vote[c][k]]
            pq[f"nearest_{c}_k{k}"] = [cls_t[j] if j >= 0 else "" for j in conf[c]["lab"][:, 0]]
        pq[f"same_neighbor_set_AB_k{k}"] = same_set.astype(int)
        pq[f"neighbor_overlap_AB_k{k}"] = overlap
        pq[f"total_variation_AB_k{k}"] = tv
        for sname, sel in (("all", common), ("patient_annotated", common & (ann == 1)), ("patient_not_annotated", common & (ann == 0))):
            e4[f"k{k}_{sname}"] = {
                "n": int(sel.sum()), "same_argmax_share": float((pred["A"][k][sel] == pred["B"][k][sel]).mean()),
                "identical_neighbor_set_share": float(same_set[sel].mean()), "mean_neighbor_overlap": float(overlap[sel].mean()),
                "mean_abs_change_top_probability": float(dtop[sel].mean()), "max_abs_change_top_probability": float(dtop[sel].max()),
                "mean_total_variation": float(tv[sel].mean()),
                "accuracy_A": float((pred["A"][k][sel] == y[sel]).mean()), "accuracy_B": float((pred["B"][k][sel] == y[sel]).mean())}
        e5[str(k)] = {c: float((pred[c][k][common] == conf[c]["lab"][common, 0]).mean()) for c in ("A", "B")}
    for c in ("A", "B"):
        m1 = metrics(y[common], conf[c]["lab"][common, 0], nc)
        e5[f"nearest_neighbor_alone_{c}"] = m1
    res["E4_per_query_A_vs_B"] = e4
    res["E5_prediction_equals_nearest_neighbor_label"] = e5

    # per-group F1
    pg = []
    for c in ("A", "B"):
        for k in KS:
            f, sup = per_class_f1(y[common], pred[c][k][common], nc)
            pg += [{"config": c, "k": k, "group": g, "n_queries": int(s), "f1": float(v)} for g, s, v in zip(cls_t, sup, f)]
    pg = pd.DataFrame(pg)

    # E6 named cases
    meta = pd.read_csv(inp["case_metadata"][0], low_memory=False, dtype=str)
    meta["sid"] = meta["Slide.Scan.File"].str.replace(r"\.0$", "", regex=True)
    e6, crow = {}, []
    full = {"S": conf_s, **conf}
    for name, sid, grp in CASES:
        ids_meta = set(meta.loc[meta["Comp.Bio.ID"] == name, "sid"])
        assert ids_meta == {sid}, (name, ids_meta)
        assert g_map[sid] == grp, (name, g_map[sid])
        entry = {"scan_id": sid, "recorded_group": grp, "patient_annotated": int(annotated[sid]), "configs": {}}
        for c, cf in full.items():
            per_k = {}
            for k in KS:
                p, ok = bayes(cf, k)
                per_k[k] = case_row(cf, p, ok, sid, k)
                if per_k[k] is not None:
                    for g, v in per_k[k]["probs"].items():
                        crow.append({"case": name, "scan_id": sid, "recorded_group": grp, "config": c, "k": k, "group": g,
                                     "probability": v, "neighbor_count": per_k[k]["counts"].get(g, 0)})
            entry["configs"][c] = {"k5": per_k[5], "k18": per_k[18],
                                   "reading_checks": case_reading(name, per_k[5], per_k[18]) if per_k[5] and per_k[18] else None}
        e6[name] = entry
    res["E6_cases"] = e6

    # pre-stated yardstick
    yard = {
        "metric_changes_within_0.010_and_ci_includes_0": bool(all(
            abs(e1[str(k)][n]["B_minus_A"]) <= 0.010 and e1[str(k)][n]["ci95"][0] <= 0 <= e1[str(k)][n]["ci95"][1]
            for k in KS for n in names)),
        "bayesian_minus_plurality_signs_unchanged": bool(all(
            np.sign(e3[f"A_k{k}"]["bayesian_minus_plurality"][n]) == np.sign(e3[f"B_k{k}"]["bayesian_minus_plurality"][n])
            for k in KS for n in names)),
        "f1_k18_within_0.005_of_B_max": bool(abs(e2["B"]["f1_k18_minus_max"]) <= 0.005),
        "all_case_reading_checks_hold_under_B": bool(all(all(e6[n]["configs"]["B"]["reading_checks"].values()) for n in e6)),
    }
    yard["little_change"] = bool(all(yard.values()))
    res["yardstick"] = yard
    res["populations"] = {"S_queries_returned_k5": int(ps[5][1].sum()), "tumor_population": len(ids_t),
                          "returned": {c: int(v.sum()) for c, v in returned.items()}, "common": int(common.sum()),
                          "not_in_common": [ids_t[i] for i in np.flatnonzero(~common)],
                          "patients_common": int(len(set(pats[common]))), "patient_annotated_common": int((ann[common] == 1).sum()),
                          "groups": cls_t}

    mt.to_csv(os.path.join(dest, "metrics_by_config.csv"), index=False)
    sw.to_csv(os.path.join(dest, "bayesian_metrics_by_k.csv"), index=False)
    pg.to_csv(os.path.join(dest, "per_group_f1.csv"), index=False)
    pd.DataFrame(crow).to_csv(os.path.join(dest, "cases.csv"), index=False)
    pq.to_csv(os.path.join(dest, "per_query.csv.gz"), index=False)
    env["seconds"] = round(time.time() - t0, 1)
    json.dump({"status": "COMPLETE", "record": "docs/plans/0002 E1-BAYHO", "parameters": {"T": T, "k": list(KS), "k_sweep": [K_SWEEP[0], K_SWEEP[-1]],
               "bootstrap_B": BOOT_B, "bootstrap_seed": BOOT_SEED}, "gates": gates, "results": res, "env": env},
              open(os.path.join(dest, "results.json"), "w"), indent=1)
    shutil.copy2(os.path.abspath(__file__), os.path.join(dest, "e1_bayes_heldout_filter_sensitivity.py"))
    if os.path.exists(test_path):
        shutil.copy2(test_path, os.path.join(dest, "test_e1_bayes_heldout_filter_sensitivity.py"))
    with open(os.path.join(dest, "SHA256SUMS.txt"), "w") as f:
        for fn in sorted(os.listdir(dest)):
            if fn != "SHA256SUMS.txt":
                f.write(f"{sha256(os.path.join(dest, fn))}  {fn}\n")
    print(json.dumps({"gates": {k: v["pass"] for k, v in gates.items()}, "yardstick": yard, "dest": dest}, indent=1))


if __name__ == "__main__":
    main()
