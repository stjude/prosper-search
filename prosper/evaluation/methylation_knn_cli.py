"""X2 (R2.3) -- are the H&E retrieval failures also methylation outliers?

For every library slide we know (a) whether the frozen primary retrieval got it
right (k=5 closest-plurality vote, top-10 hit) and (b) its sample's DNA
methylation profile (EPIC 850K, SWAN-normalised, top-20K most variable CpGs;
``COMET_top20K_4735samples_swan.csv``, one column per Sentrix id). This module
computes, per slide, the **methylation 10-NN neighbourhood** of its sample among
the other library samples (same patient and same sample excluded, mirroring the
retrieval protocol) and reports how the two failure sets relate:

* methylation label purity -- the fraction of the 10 nearest samples that carry
  the slide's tumour group -- for H&E-correct vs H&E-failed queries;
* the methylation k-NN vote's own accuracy, and the concordance table
  (H&E wrong x methylation wrong) with relative risk;
* AUROC of (1 - purity) as a predictor of H&E failure;
* everything with patient-clustered bootstrap CIs (B, seed as the paper).

The reading it supports: if failed H&E queries are, as a group, also atypical
in methylation space, part of the retrieval error is biological ambiguity of the
case rather than a deficiency of the image representation; if they are not, the
failures are image-side. No claim is made about either until the numbers exist.

Distance between samples is 1 - Pearson correlation over the CpG panel (the
convention of methylation-classifier work); Euclidean is available as a check.
Slides sharing a sample share a profile, so neighbours are found at the sample
level and each slide inherits its sample's neighbourhood.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .external_cli import vote

AMBIG = "__AMBIGUOUS__"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[2],
                                       text=True).strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


def load_beta(path: Path, sentrix_ids, nrows=None) -> pd.DataFrame:
    """CpG x sample matrix restricted to ``sentrix_ids`` (samples x CpGs on return)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    first = header[0]
    keep = [c for c in header[1:] if c in set(sentrix_ids)]
    df = pd.read_csv(path, usecols=[first] + keep, index_col=0, nrows=nrows,
                     dtype={c: np.float32 for c in keep})
    return df.T                                          # samples x CpGs


def sample_distances(X: pd.DataFrame, metric: str) -> pd.DataFrame:
    A = X.to_numpy(dtype=np.float64)
    # CpG-wise mean imputation of missing betas (few in the SWAN-filtered panel)
    col_mean = np.nanmean(A, axis=0)
    nan = np.isnan(A)
    if nan.any():
        A[nan] = np.take(col_mean, np.where(nan)[1])
    if metric == "correlation":
        Z = A - A.mean(axis=1, keepdims=True)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        D = 1.0 - Z @ Z.T
    elif metric == "euclidean":
        sq = (A * A).sum(axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0.0))
    else:
        raise ValueError(metric)
    np.fill_diagonal(D, np.inf)
    return pd.DataFrame(D, index=X.index, columns=X.index)


def he_outcomes(ranked_lists: Path, k: int, depth: int) -> pd.DataFrame:
    r = pd.read_csv(ranked_lists, dtype={"query": str, "neighbour": str})
    rows = []
    for q, g in r.sort_values(["query", "rank"]).groupby("query", sort=False):
        labs = g["neighbour_label"].tolist()
        truth = g["true_label"].iloc[0]
        pred, tied, share, ent, _ = vote(labs[:k])
        rows.append({"slide": q, "true_label": truth, "he_pred": pred,
                     "he_wrong": pred != truth,
                     "he_top1_hit": labs[0] == truth,
                     "he_miss_depth": truth not in labs[:depth],
                     "he_vote_share": share, "he_vote_entropy": ent})
    return pd.DataFrame(rows).set_index("slide")


def meth_neighbourhood(D: pd.DataFrame, sample_meta: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per sample: k nearest other samples (same patient excluded), purity, vote."""
    labels = sample_meta["label"]
    patients = sample_meta["patient"]
    ids = D.index.tolist()
    lab = labels.reindex(ids).to_numpy()
    pat = patients.reindex(ids).to_numpy()
    Dv = D.to_numpy()
    out = []
    for i, s in enumerate(ids):
        d = Dv[i].copy()
        d[pat == pat[i]] = np.inf                        # same patient (incl. self)
        order = np.argsort(d, kind="stable")[:k]
        nl = lab[order].tolist()
        pred, tied, share, ent, _ = vote(nl)
        purity = float(np.mean([x == lab[i] for x in nl]))
        out.append({"sample": s, "meth_pred": pred, "meth_wrong": pred != lab[i],
                    "meth_purity": purity, "meth_top1_dist": float(d[order[0]]),
                    "meth_top1_label": nl[0], "meth_top1_hit": nl[0] == lab[i],
                    "meth_vote_share": share, "meth_vote_entropy": ent,
                    "meth_neighbours": ";".join(ids[j] for j in order)})
    return pd.DataFrame(out).set_index("sample")


def _auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC of ``score`` for ``y`` (1 = positive), ties averaged."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarise(df: pd.DataFrame) -> dict:
    w, c = df[df.he_wrong], df[~df.he_wrong]
    p_meth_wrong_given_he_wrong = float(w.meth_wrong.mean()) if len(w) else float("nan")
    p_meth_wrong_given_he_right = float(c.meth_wrong.mean()) if len(c) else float("nan")
    return {
        "n": int(len(df)), "n_he_wrong": int(df.he_wrong.sum()),
        "he_accuracy": float(1 - df.he_wrong.mean()),
        "meth_knn_accuracy": float(1 - df.meth_wrong.mean()),
        "meth_purity_mean_he_wrong": float(w.meth_purity.mean()) if len(w) else float("nan"),
        "meth_purity_mean_he_right": float(c.meth_purity.mean()) if len(c) else float("nan"),
        "meth_purity_median_he_wrong": float(w.meth_purity.median()) if len(w) else float("nan"),
        "meth_purity_median_he_right": float(c.meth_purity.median()) if len(c) else float("nan"),
        "purity_gap": float(c.meth_purity.mean() - w.meth_purity.mean()) if len(w) and len(c) else float("nan"),
        "p_meth_wrong_given_he_wrong": p_meth_wrong_given_he_wrong,
        "p_meth_wrong_given_he_right": p_meth_wrong_given_he_right,
        "relative_risk": (p_meth_wrong_given_he_wrong / p_meth_wrong_given_he_right
                          if p_meth_wrong_given_he_right else float("nan")),
        "auroc_impurity_for_he_failure": _auroc(1 - df.meth_purity.to_numpy(),
                                                df.he_wrong.to_numpy().astype(int)),
        "frac_he_wrong_with_meth_purity_le_0.5": float((w.meth_purity <= 0.5).mean()) if len(w) else float("nan"),
        "frac_he_right_with_meth_purity_le_0.5": float((c.meth_purity <= 0.5).mean()) if len(c) else float("nan"),
    }


def patient_bootstrap(df: pd.DataFrame, B: int, seed: int, keys) -> dict:
    rng = np.random.default_rng(seed)
    groups = {p: g.index.to_numpy() for p, g in df.groupby("patient")}
    pids = np.array(list(groups))
    acc = {k: [] for k in keys}
    for _ in range(B):
        take = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([groups[p] for p in take])
        s = summarise(df.loc[idx])
        for k in keys:
            acc[k].append(s[k])
    return {k: {"lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5)),
                "B": B} for k, v in acc.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--beta", required=True, type=Path)
    p.add_argument("--coverage", required=True, type=Path,
                   help="CSV with Slide.Scan.File, Sample.ID, UID.Subject, Tumor.group, sx (Sentrix)")
    p.add_argument("--ranked-lists", required=True, type=Path)
    p.add_argument("--k", type=int, default=10, help="methylation neighbourhood size")
    p.add_argument("--he-k", type=int, default=5, help="H&E vote depth (primary k)")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--metric", choices=["correlation", "euclidean"], default="correlation")
    p.add_argument("--drop-labels", default="MISC")
    p.add_argument("--iter", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--nrows", type=int, default=None, help="CpG rows to read (smoke tests)")
    p.add_argument("--task", default="x2_methylation")
    return p


def run(a) -> dict:
    t0 = time.time()
    out_dir = paths.output_root() / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    cov = pd.read_csv(a.coverage, dtype=str)
    drop = [x for x in a.drop_labels.split(",") if x]
    cov = cov[~cov["Tumor.group"].isin(drop)].dropna(subset=["sx"])
    cov = cov.rename(columns={"Slide.Scan.File": "slide", "UID.Subject": "patient",
                              "Tumor.group": "label", "Sample.ID": "sample_id"})
    sample_meta = (cov.groupby("sx").agg(label=("label", "first"), patient=("patient", "first"),
                                         n_slides=("slide", "size")))
    multi = cov.groupby("sx")["label"].nunique()
    if (multi > 1).any():
        raise SystemExit(f"{int((multi > 1).sum())} Sentrix ids map to >1 tumour group")
    X = load_beta(a.beta, sample_meta.index, nrows=a.nrows)
    missing = sorted(set(sample_meta.index) - set(X.index))
    sample_meta = sample_meta.loc[[s for s in sample_meta.index if s in X.index]]
    X = X.loc[sample_meta.index]
    D = sample_distances(X, a.metric)
    nb = meth_neighbourhood(D, sample_meta, a.k)
    he = he_outcomes(a.ranked_lists, a.he_k, a.depth)
    per = cov.set_index("slide")[["sample_id", "sx", "patient", "label"]].join(he, how="inner")
    per = per.join(nb, on="sx", how="inner")
    per.to_csv(out_dir / f"per_slide_{a.task}.csv")
    keys = ["meth_knn_accuracy", "purity_gap", "p_meth_wrong_given_he_wrong",
            "p_meth_wrong_given_he_right", "relative_risk", "auroc_impurity_for_he_failure"]
    overall = summarise(per)
    ci = patient_bootstrap(per, a.iter, a.seed, keys)
    per_class = {lab: summarise(g) for lab, g in per.groupby("label")}
    # top-10-miss variant of the H&E failure definition
    per_miss = per.assign(he_wrong=per.he_miss_depth)
    miss = summarise(per_miss)
    miss_ci = patient_bootstrap(per_miss, a.iter, a.seed, keys)
    res = {
        "config": {"beta": str(a.beta), "coverage": str(a.coverage),
                   "ranked_lists": str(a.ranked_lists), "k_methylation": a.k,
                   "he_k": a.he_k, "depth": a.depth, "metric": a.metric,
                   "drop_labels": drop, "bootstrap_B": a.iter, "seed": a.seed,
                   "bootstrap": "patient-cluster resampling (all slides of a resampled patient)",
                   "n_cpg": int(X.shape[1]), "nrows": a.nrows, "prosper_commit": _commit()},
        "n_samples": int(len(sample_meta)), "n_slides": int(len(per)),
        "sentrix_ids_missing_from_beta": missing,
        "overall_he_wrong_at_k": overall, "overall_ci": ci,
        "he_miss_at_depth": miss, "he_miss_ci": miss_ci,
        "per_class": per_class,
        "meth_confusion_top_pairs": Counter(
            (t, p) for t, p in zip(per.label, per.meth_pred) if t != p).most_common(10),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / f"{a.task}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    return res


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    res = run(a)
    o = res["overall_he_wrong_at_k"]
    print(f"n={res['n_slides']} slides / {res['n_samples']} samples, CpGs={res['config']['n_cpg']}")
    print(f"H&E acc {o['he_accuracy']:.4f}  methylation kNN acc {o['meth_knn_accuracy']:.4f}")
    print(f"purity: H&E-right {o['meth_purity_mean_he_right']:.3f} vs H&E-wrong {o['meth_purity_mean_he_wrong']:.3f}"
          f"  gap {o['purity_gap']:+.3f} {res['overall_ci']['purity_gap']}")
    print(f"P(meth wrong | H&E wrong) {o['p_meth_wrong_given_he_wrong']:.3f} vs | right {o['p_meth_wrong_given_he_right']:.3f}"
          f"  RR {o['relative_risk']:.2f} {res['overall_ci']['relative_risk']}")
    print(f"AUROC(impurity -> H&E failure) {o['auroc_impurity_for_he_failure']:.3f} {res['overall_ci']['auroc_impurity_for_he_failure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
