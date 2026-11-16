"""C1d: are retrieval results consistent across independent slides from the same patient?

Question: do separate slides of the same patient give consistent
retrieval results? With same-patient slides masked
out of every query's library ([decision record]), two slides of one patient are
retrieved against *identical* candidate sets minus themselves, so agreement
between their results is a clean stability measurement rather than leakage.

For every within-patient pair in the evaluated population (477 patients with
>1 slide; 1,898 pairs after the MISC drop) this reports:

* top-1 neighbour label agreement, top-5 plurality label agreement;
* Jaccard overlap of the top-5 neighbour *slide* sets and of the top-5 label sets;
* Jensen-Shannon divergence between the Bayesian posteriors (N=5 and N=18,
  T=20 -- the published Tables S11/S13 settings, computed with the module that
  reproduces those tables to 5.5e-15);
* whether the two slides are concordant on hit@5 / generous-correct@5.

Every statistic is paired with a **class-matched null**: for each same-patient
pair (a, b), a slide c of a's class from a *different* patient is drawn and the
same statistics are computed for (a, c). "Same-patient slides agree more than
class-matched strangers" is the claim R4.a's second ask can support; without
the null, agreement would just restate that both slides are, say, RMS.
Uncertainty: percentile bootstrap over patients (resampled with replacement).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import paths
from .cli import PUBLISHED, PUBLISHED_DROP_LABELS, _git_commit
from .slide_eval import filter_similarity_matrix, load_metadata, load_similarity_matrix
from ..bayesian.disease_probs import compute_disease_probs


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _plurality(labels: List[str]) -> str:
    c = Counter(labels); top = max(c.values())
    tied = {l for l, n in c.items() if n == top}
    return next(l for l in labels if l in tied)


def _pair_stats(a: str, b: str, R: Dict, post: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    ra, rb = R[a], R[b]
    ja = len(set(ra["ids5"]) & set(rb["ids5"])) / max(1, len(set(ra["ids5"]) | set(rb["ids5"])))
    jl = len(set(ra["labs5"]) & set(rb["labs5"])) / max(1, len(set(ra["labs5"]) | set(rb["labs5"])))
    out = {"top1_label_agree": float(ra["top1"] == rb["top1"]),
           "top5_plurality_agree": float(ra["plur5"] == rb["plur5"]),
           "jaccard_top5_ids": ja, "jaccard_top5_labels": jl,
           "both_hit5": float(ra["hit5"] and rb["hit5"]),
           "split_hit5": float(ra["hit5"] != rb["hit5"]),
           "both_generous5": float(ra["gen5"] and rb["gen5"])}
    for N, P in post.items():
        out[f"jsd_posterior_N{N}"] = _jsd(P[a], P[b])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="uni2")
    p.add_argument("--method", default="ot_cosine")
    p.add_argument("--tumor_only", action="store_true")
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--n-neighbors", type=int, nargs="+", default=[5, 18])
    p.add_argument("--iter", type=int, default=2000, dest="B")
    p.add_argument("--seed", type=int, default=PUBLISHED["seed"])
    p.add_argument("--task", default="c1d_patient_consistency")
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
    pat = dict(zip(meta_f[PUBLISHED["id_col"]].astype(str), meta_f[PUBLISHED["patient_col"]].astype(str)))
    arr = sim_f.to_numpy(dtype=float)
    arr = np.where(np.isnan(arr), 1e6, arr)
    col_labels = [lab[i] for i in ids]

    # Ranked neighbours (sentinel-safe: sentinels sort last and are never reached at depth 10)
    R = {}
    for i, q in enumerate(ids):
        order = np.argsort(arr[i], kind="mergesort")
        top = [ids[j] for j in order[:10] if arr[i, j] < 1e6]
        labs = [lab[j] for j in top]
        R[q] = {"top1": labs[0] if labs else None, "ids5": top[:5], "labs5": labs[:5],
                "plur5": _plurality(labs[:5]) if labs else None,
                "hit5": lab[q] in labs[:5], "gen5": bool(labs[:5]) and lab[q] in {
                    l for l, n in Counter(labs[:5]).items() if n == max(Counter(labs[:5]).values())}}

    post = {}
    for N in a.n_neighbors:
        _, df = compute_disease_probs(arr, col_labels, n_neighbors=N, T=a.temperature,
                                      test_slide_ids=ids)
        post[N] = {q: df.loc[q].to_numpy(dtype=float) for q in ids}

    by_pat = defaultdict(list)
    for q in ids:
        by_pat[pat[q]].append(q)
    multi = {k: v for k, v in by_pat.items() if len(v) > 1}
    by_class = defaultdict(list)
    for q in ids:
        by_class[lab[q]].append(q)
    rng = np.random.default_rng(a.seed)

    rows = []
    for pid, slides in multi.items():
        for x, y in combinations(sorted(slides), 2):
            s = _pair_stats(x, y, R, post)
            s.update({"patient": pid, "a": x, "b": y, "class_a": lab[x], "class_b": lab[y],
                      "same_class": float(lab[x] == lab[y]), "kind": "same_patient"})
            rows.append(s)
            # class-matched stranger for the null
            cands = [c for c in by_class[lab[x]] if pat[c] != pid]
            if cands:
                c = cands[rng.integers(len(cands))]
                n = _pair_stats(x, c, R, post)
                n.update({"patient": pid, "a": x, "b": c, "class_a": lab[x], "class_b": lab[c],
                          "same_class": 1.0, "kind": "class_matched_null"})
                rows.append(n)
    df = pd.DataFrame(rows)
    stat_cols = [c for c in df.columns if c not in ("patient", "a", "b", "class_a", "class_b", "kind")]

    def _means(d: pd.DataFrame) -> Dict[str, float]:
        return {c: float(d[c].mean()) for c in stat_cols}

    same = df[df.kind == "same_patient"]; null = df[df.kind == "class_matched_null"]
    point = {"same_patient": _means(same), "class_matched_null": _means(null),
             "difference": {c: float(same[c].mean() - null[c].mean()) for c in stat_cols}}
    # bootstrap over patients
    pats = sorted(multi)
    draws = {c: np.empty(a.B) for c in stat_cols}
    g_same = {k: v for k, v in same.groupby("patient")}
    g_null = {k: v for k, v in null.groupby("patient")}
    for b in range(a.B):
        pick = rng.choice(pats, size=len(pats), replace=True)
        s_ = pd.concat([g_same[k] for k in pick if k in g_same])
        n_ = pd.concat([g_null[k] for k in pick if k in g_null])
        for c in stat_cols:
            draws[c][b] = s_[c].mean() - n_[c].mean()
    ci = {c: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for c, v in draws.items()}

    results = {"config": {"model": a.model, "method": a.method, "tumor_only": a.tumor_only,
                          "matrix": str(sim_path), "temperature": a.temperature,
                          "n_neighbors": a.n_neighbors, "bootstrap_B": a.B, "seed": a.seed,
                          "prosper_commit": _git_commit(), "lsf_job": os.environ.get("LSB_JOBID")},
               "n_queries": len(ids), "n_patients_multi_slide": len(multi),
               "n_same_patient_pairs": int(len(same)), "n_null_pairs": int(len(null)),
               "same_class_fraction_within_patient": float(same["same_class"].mean()),
               "point": point, "difference_ci_over_patients": ci,
               "jsd_units": "nats; 0 identical, ln 2 = 0.693 maximal"}
    out = paths.result_json(a.task, f"consistency_{a.model}_{a.method}{'_tumor_only' if a.tumor_only else ''}")
    out.write_text(json.dumps(results, indent=2, default=str))
    df.to_csv(out.parent / f"pairs_{a.model}_{a.method}.csv", index=False)
    print(f"[c1d] wrote {out}")
    for c in stat_cols:
        print(f"[c1d] {c:<24} same-patient {point['same_patient'][c]:.4f}  null {point['class_matched_null'][c]:.4f}  "
              f"Δ {point['difference'][c]:+.4f} [{ci[c][0]:+.4f}, {ci[c][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
