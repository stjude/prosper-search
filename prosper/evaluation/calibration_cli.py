"""C10: is the Bayesian posterior "calibrated", and what does abstaining on it buy? (E5, R2.8)

The abstract calls the tumour-group probabilities "calibrated"; the paper
contains no reliability analysis, and the Methods assume uniform class priors,
which decalibrates against prevalence by construction. This computes, on the
patient-masked matrix and the frozen population:

* **Reliability**: top-label expected calibration error (ECE, 15 equal-width
  bins and 15 equal-mass bins), maximum calibration error, per-bin table;
  multiclass Brier score, negative log-likelihood; Brier skill vs a
  prevalence-only predictor (class frequencies in the library) -- the honest
  baseline given the uniform-prior assumption.
* **Risk-coverage** (R2.8's abstention framing): selective accuracy when the
  system abstains below a confidence threshold, thresholding on max posterior
  and on the top-2 likelihood ratio (the ``LR1t2`` column Tables S10/S12 already
  carry), with the area under the risk-coverage curve (AURC).
* **Entropy of the posterior for correct vs incorrect predictions** (C6's
  entropy arm, R2.3): means, and AUROC of entropy as an error detector.

Posteriors come from ``prosper.bayesian.disease_probs`` (reproduces Tables
S11/S13 to 5.5e-15) at N=5 and N=18, T=20. CIs: patient-clustered PPS
bootstrap on ECE, Brier and accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import paths, resample
from .cli import PUBLISHED, PUBLISHED_DROP_LABELS, _git_commit
from .slide_eval import filter_similarity_matrix, load_metadata, load_similarity_matrix
from ..bayesian.disease_probs import compute_disease_probs


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15, equal_mass: bool = False):
    if equal_mass:
        edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    else:
        edges = np.linspace(0, 1, bins + 1); edges[-1] = 1.0 + 1e-9
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, bins - 1)
    e = 0.0; mce = 0.0; table = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            table.append({"bin": b, "n": 0}); continue
        c_mean, a_mean = float(conf[m].mean()), float(correct[m].mean())
        gap = abs(c_mean - a_mean)
        e += gap * m.mean(); mce = max(mce, gap)
        table.append({"bin": b, "n": int(m.sum()), "confidence": c_mean, "accuracy": a_mean, "gap": gap})
    return float(e), float(mce), table


def risk_coverage(score: np.ndarray, correct: np.ndarray) -> Dict:
    order = np.argsort(-score, kind="mergesort")
    err = 1.0 - correct[order]
    cum_err = np.cumsum(err) / np.arange(1, len(err) + 1)
    cov = np.arange(1, len(err) + 1) / len(err)
    aurc = float(getattr(np, "trapezoid", getattr(np, "trapz", None))(cum_err, cov))
    pts = {}
    for c in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        n = max(1, int(round(c * len(err))))
        pts[f"coverage_{c}"] = {"selective_accuracy": float(1.0 - cum_err[n - 1]),
                               "threshold": float(score[order][n - 1])}
    return {"aurc": aurc, "points": pts}


def _auroc(score: np.ndarray, positive: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if positive.all() or (~positive).all():
        return float("nan")
    return float(roc_auc_score(positive.astype(int), score))


def analyse(P: pd.DataFrame, y: np.ndarray, prior: np.ndarray) -> Dict:
    classes = list(P.columns)
    probs = P.to_numpy(dtype=float)
    yi = np.array([classes.index(t) for t in y])
    pred = probs.argmax(axis=1)
    correct = (pred == yi).astype(float)
    conf = probs.max(axis=1)
    srt = np.sort(probs, axis=1)
    lr12 = np.log(np.clip(srt[:, -1], 1e-12, 1)) - np.log(np.clip(srt[:, -2], 1e-12, 1))
    onehot = np.zeros_like(probs); onehot[np.arange(len(yi)), yi] = 1
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    brier_prior = float(np.mean(np.sum((prior[None, :] - onehot) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(yi)), yi], 1e-12, 1))))
    ent = -(np.clip(probs, 1e-12, 1) * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)
    ece_w, mce_w, tab_w = ece(conf, correct, 15, False)
    ece_m, mce_m, tab_m = ece(conf, correct, 15, True)
    return {
        "accuracy": float(correct.mean()),
        "ece_15_equal_width": ece_w, "mce_15_equal_width": mce_w, "reliability_equal_width": tab_w,
        "ece_15_equal_mass": ece_m, "mce_15_equal_mass": mce_m, "reliability_equal_mass": tab_m,
        "brier_multiclass": brier, "brier_prevalence_prior": brier_prior,
        "brier_skill_score": float(1.0 - brier / brier_prior) if brier_prior > 0 else None,
        "nll": nll,
        "mean_confidence": float(conf.mean()),
        "risk_coverage_max_posterior": risk_coverage(conf, correct),
        "risk_coverage_top2_log_lr": risk_coverage(lr12, correct),
        "entropy": {"mean_correct": float(ent[correct == 1].mean()),
                    "mean_incorrect": float(ent[correct == 0].mean()) if (correct == 0).any() else None,
                    "auroc_entropy_detects_error": _auroc(ent, correct == 0),
                    "auroc_confidence_detects_correct": _auroc(conf, correct == 1),
                    "units": "nats; uniform over 15 classes = 2.708"},
        "_per_query": {"conf": conf, "correct": correct, "brier": np.sum((probs - onehot) ** 2, axis=1)},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="uni2")
    p.add_argument("--method", default="ot_cosine")
    p.add_argument("--tumor_only", action="store_true")
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--n-neighbors", type=int, nargs="+", default=[5, 18])
    p.add_argument("--iter", type=int, default=2000, dest="B")
    p.add_argument("--seed", type=int, default=PUBLISHED["seed"])
    p.add_argument("--task", default="c10_calibration")
    a = p.parse_args(argv)

    sim_path = paths.require(paths.similarity_matrix(a.model, a.method, tumor_only=a.tumor_only,
                                                     no_same_patient=True), "matrix")
    sim = load_similarity_matrix(str(sim_path))
    meta = load_metadata(str(paths.slide_metadata()), label_col=PUBLISHED["label_col"],
                         id_col=PUBLISHED["id_col"], patient_col=PUBLISHED["patient_col"],
                         drop_labels=list(PUBLISHED_DROP_LABELS[bool(a.tumor_only)]),
                         min_class_count=None)
    sim_f, meta_f = filter_similarity_matrix(sim, meta, id_col=PUBLISHED["id_col"])
    ids = [str(i) for i in sim_f.index]
    lab = dict(zip(meta_f[PUBLISHED["id_col"]].astype(str), meta_f[PUBLISHED["label_col"]].astype(str)))
    arr = sim_f.to_numpy(dtype=float)
    arr = np.where(np.isnan(arr), 1e6, arr)
    col_labels = [lab[i] for i in ids]
    y = np.array([lab[i] for i in ids], dtype=object)
    classes = sorted(set(col_labels))
    cnt = Counter(col_labels)
    prior = np.array([cnt[c] / len(col_labels) for c in classes])

    index = resample.build_patient_index(meta_f, PUBLISHED["id_col"], PUBLISHED["label_col"],
                                         PUBLISHED["patient_col"])
    idx_of = {q: i for i, q in enumerate(ids)}
    results = {"config": {"model": a.model, "method": a.method, "tumor_only": a.tumor_only,
                          "matrix": str(sim_path), "temperature": a.temperature,
                          "n_neighbors": a.n_neighbors, "bootstrap_B": a.B, "seed": a.seed,
                          "n_queries": len(ids), "classes": classes,
                          "prior_note": "Methods assume uniform class priors; the prevalence "
                                        "prior here is the library class frequency",
                          "prosper_commit": _git_commit(), "lsf_job": os.environ.get("LSB_JOBID")},
               "by_N": {}}
    for N in a.n_neighbors:
        _, P = compute_disease_probs(arr, col_labels, n_neighbors=N, T=a.temperature,
                                     test_slide_ids=ids)
        P = P[classes]
        r = analyse(P, y, prior)
        pq = r.pop("_per_query")
        rng = np.random.default_rng(a.seed)
        draws = {"ece_15_equal_width": [], "brier_multiclass": [], "accuracy": []}
        for b in range(a.B):
            s = resample.sample_queries(index, classes, rng, mode=PUBLISHED["bootstrap_mode"])
            rows = np.array([idx_of[x] for x in s if x in idx_of], dtype=int)
            if rows.size == 0:
                continue
            e, _, _ = ece(pq["conf"][rows], pq["correct"][rows], 15, False)
            draws["ece_15_equal_width"].append(e)
            draws["brier_multiclass"].append(float(pq["brier"][rows].mean()))
            draws["accuracy"].append(float(pq["correct"][rows].mean()))
        r["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items() if v}
        results["by_N"][str(N)] = r
        print(f"[c10] N={N}: acc {r['accuracy']:.4f}  ECE {r['ece_15_equal_width']:.4f} "
              f"[{r['ci']['ece_15_equal_width'][0]:.4f},{r['ci']['ece_15_equal_width'][1]:.4f}]  "
              f"Brier {r['brier_multiclass']:.4f} (prior {r['brier_prevalence_prior']:.4f})  "
              f"AURC {r['risk_coverage_max_posterior']['aurc']:.4f}  "
              f"entropy correct/incorrect {r['entropy']['mean_correct']:.3f}/{r['entropy']['mean_incorrect']:.3f}  "
              f"AUROC(entropy→error) {r['entropy']['auroc_entropy_detects_error']:.3f}")
    out = paths.result_json(a.task, f"calibration_{a.model}_{a.method}{'_tumor_only' if a.tumor_only else ''}")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[c10] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
