"""External-cohort retrieval evaluation against the COMET library.

Tasks C3 (CCDI re-analysis, R3.2), C12 (the ``solid_tumor_validation`` cohort,
R1.3 / R2.7) and any future external set share one need the published code
never had as a reusable piece: rank a *rectangular* query x library distance
block, vote, and score -- with or without query labels.

Two modes, one code path:

* **Label-free** (what an external cohort with no manifest yet allows): per
  query, the ranked neighbours, the plurality label at each k, its vote share,
  the vote entropy, and the top-1 / mean top-5 distances. Cohort-level: the
  predicted-label distribution, and the distance / vote-share / entropy
  distributions side by side with the same statistics for COMET's own queries
  (``--internal-reference-matrix``), which is the honest *domain-shift*
  diagnostic for a different-scanner cohort before any label is seen.
* **Labelled** (``--query-metadata``): hit@k, precision@k, recall@k, AP@k,
  nDCG@k, MRR (``metrics.compute_all``), the three voting F1s (generous /
  strict-majority / closest) with per-class breakdown, and patient-clustered
  percentile bootstrap CIs (PPS, the frozen protocol's design; a query with no
  patient id is its own cluster, which reduces to i.i.d. within class).
  Queries whose true label is not a library class are reported as an
  **out-of-library** bucket -- what they retrieve, how confidently -- rather
  than being scored as errors, because a class absent from the library cannot
  be retrieved by any system (the G4 lesson).

Library and vocabulary follow the frozen protocol: the COMET all-tissue
population (``drop_labels=['MISC']``, 15 classes incl. NORMAL), in the matrix's
own column order ([decision record]). Sentinel / NaN cells are skipped, not trusted to
sort last.

Known-answer self-check (``--assert-known-answer``): run with the COMET masked
square matrix as both query block and library, with COMET metadata as the
query metadata. The self and same-patient cells are sentinels, so the result
must reproduce C2's ``metrics_uni2_ot_cosine.json`` hit@k / precision@k and
G2's generous macro F1 at k=5 (0.791156) exactly. Nothing external is believed
before that passes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from . import metrics, paths, resample
from .cli import PUBLISHED, PUBLISHED_DROP_LABELS, _git_commit
from .slide_eval import load_metadata, load_similarity_matrix

SENTINEL = 1.0e6
AMBIGUOUS = "Ambiguous"


# ---------------------------------------------------------------------------
# Ranking and voting
# ---------------------------------------------------------------------------

def rank_block(M: pd.DataFrame, lib_ids: List[str], lib_labels: Dict[str, str],
               lib_patients: Dict[str, str], q_ids: List[str],
               q_patients: Optional[Dict[str, str]], depth: int):
    """Top-``depth`` library neighbours per query row.

    Skips sentinel / NaN cells, the query's own id if it is in the library, and
    (self-check mode only) library slides of the query's own patient.
    """
    arr = M.loc[q_ids, lib_ids].to_numpy(dtype=float)
    n = len(q_ids)
    nbr_ids = np.empty((n, depth), dtype=object)
    nbr_labels = np.empty((n, depth), dtype=object)
    nbr_dists = np.full((n, depth), np.nan, dtype=float)
    short = 0
    for i in range(n):
        row = arr[i]
        order = np.argsort(row, kind="mergesort")
        qp = q_patients.get(q_ids[i]) if q_patients else None
        taken = 0
        for j in order:
            v = row[j]
            if not np.isfinite(v) or v >= SENTINEL:
                continue
            oid = lib_ids[j]
            if oid == q_ids[i]:
                continue
            if qp is not None and lib_patients.get(oid) == qp:
                continue
            nbr_ids[i, taken] = oid
            nbr_labels[i, taken] = lib_labels[oid]
            nbr_dists[i, taken] = v
            taken += 1
            if taken == depth:
                break
        if taken < depth:
            short += 1
            for t in range(taken, depth):
                nbr_labels[i, t] = "__NONE__"
    return nbr_ids, nbr_labels, nbr_dists, short


def vote(labels_k: List[str]) -> Tuple[str, List[str], float, float, int]:
    """Plurality over an ordered top-k.

    Returns (closest-plurality label, tied argmax set, vote share, vote entropy
    in nats normalised by ln(#distinct), #distinct labels). ``closest`` picks the
    tied label that appears earliest in rank order -- ``generous_majority``
    differs only in crediting the truth whenever it sits in the tied set.
    """
    labs = [l for l in labels_k if l != "__NONE__"]
    if not labs:
        return AMBIGUOUS, [], 0.0, 0.0, 0
    counts = Counter(labs)
    top = max(counts.values())
    tied = [l for l, c in counts.items() if c == top]
    closest = next(l for l in labs if l in tied)
    share = top / len(labs)
    p = np.array(list(counts.values()), dtype=float) / len(labs)
    ent = float(-(p * np.log(p)).sum())
    ent_norm = ent / math.log(len(counts)) if len(counts) > 1 else 0.0
    return closest, tied, float(share), float(ent_norm), len(counts)


def strict_majority(labels_k: List[str], k: int) -> str:
    labs = [l for l in labels_k if l != "__NONE__"]
    if not labs:
        return AMBIGUOUS
    need = k // 2 + 1
    counts = Counter(labs)
    reached = [l for l, c in counts.items() if c >= need]
    return reached[0] if len(reached) == 1 else AMBIGUOUS


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _f1_block(y_true: List[str], y_pred: List[str], labels_order: List[str]) -> Dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels_order, zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", labels=labels_order, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=labels_order, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels_order, zero_division=0)),
        "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else float("nan"),
        "per_class_f1": {c: float(f1_score([t == c for t in y_true], [p == c for p in y_pred], zero_division=0))
                         for c in labels_order},
    }


def per_query_frame(q_ids, q_true, nbr_ids, nbr_labels, nbr_dists, k_values, depth):
    rows = []
    for i, qid in enumerate(q_ids):
        labs = list(nbr_labels[i])
        r = {"query": qid, "true_label": q_true[i] if q_true is not None else None,
             "top1_id": nbr_ids[i, 0], "top1_label": labs[0],
             "top1_distance": float(nbr_dists[i, 0]) if np.isfinite(nbr_dists[i, 0]) else None,
             "mean_top5_distance": float(np.nanmean(nbr_dists[i, :5])),
             "n_distinct_top10": vote(labs[:10])[4]}
        for k in k_values:
            closest, tied, share, ent, _ = vote(labs[:k])
            r[f"pred_closest_k{k}"] = closest
            r[f"tied_set_k{k}"] = "|".join(tied)
            r[f"vote_share_k{k}"] = share
            r[f"vote_entropy_k{k}"] = ent
            r[f"pred_absolute_k{k}"] = strict_majority(labs[:k], k)
            if q_true is not None:
                t = q_true[i]
                r[f"hit_k{k}"] = float(t in labs[:k])
                r[f"generous_correct_k{k}"] = float(t in tied)
                r[f"pred_generous_k{k}"] = t if t in tied else closest
        rows.append(r)
    return pd.DataFrame(rows)


def distribution_block(df: pd.DataFrame, k_values) -> Dict:
    out = {"n_queries": int(len(df))}
    for k in k_values:
        col = f"pred_closest_k{k}"
        out[f"predicted_label_distribution_k{k}"] = df[col].value_counts().to_dict()
        out[f"vote_share_k{k}"] = _summ(df[f"vote_share_k{k}"])
        out[f"vote_entropy_k{k}"] = _summ(df[f"vote_entropy_k{k}"])
        out[f"fraction_ambiguous_strict_k{k}"] = float((df[f"pred_absolute_k{k}"] == AMBIGUOUS).mean())
    out["top1_distance"] = _summ(df["top1_distance"])
    out["mean_top5_distance"] = _summ(df["mean_top5_distance"])
    out["n_distinct_top10"] = _summ(df["n_distinct_top10"])
    return out


def _summ(s: pd.Series) -> Dict:
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max())}


def bootstrap(df: pd.DataFrame, q_meta: pd.DataFrame, id_col: str, label_col: str,
              patient_col: str, k_values, labels_order, B: int, seed: int,
              mode: str) -> Dict:
    """Patient-clustered PPS percentile CIs on the labelled, in-library queries."""
    index = resample.build_patient_index(q_meta, id_col, label_col, patient_col)
    classes = sorted(q_meta[label_col].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    idx_of = {q: i for i, q in enumerate(df["query"].tolist())}
    tl = df["true_label"].astype(str).to_numpy(dtype=object)
    cols = {}
    for k in k_values:
        cols[f"macro_hit_at_{k}"] = df[f"hit_k{k}"].to_numpy(dtype=float)
    draws = {name: np.empty(B) for name in cols}
    f1_draws = {f"{rule}_macro_f1_k{k}": np.empty(B) for k in k_values
                for rule in ("generous", "absolute", "closest")}
    preds = {f"{rule}_k{k}": df[f"pred_{rule}_k{k}"].astype(str).to_numpy(dtype=object)
             for k in k_values for rule in ("generous", "absolute", "closest")}
    for b in range(B):
        sampled = resample.sample_queries(index, classes, rng, mode=mode)
        rows = np.array([idx_of[s] for s in sampled if s in idx_of], dtype=int)
        if rows.size == 0:
            for d in draws.values():
                d[b] = np.nan
            for d in f1_draws.values():
                d[b] = np.nan
            continue
        rl = tl[rows]
        for name, vals in cols.items():
            per_class = [float(np.mean(vals[rows][rl == c])) for c in classes if (rl == c).any()]
            draws[name][b] = float(np.mean(per_class)) if per_class else np.nan
        yt = list(rl)
        for key, pv in preds.items():
            rule, kk = key.split("_k")
            f1_draws[f"{rule}_macro_f1_k{kk}"][b] = f1_score(
                yt, list(pv[rows]), average="macro", labels=labels_order, zero_division=0)
    out = {}
    for name, d in {**draws, **f1_draws}.items():
        v = d[~np.isnan(d)]
        out[name] = {"lo": float(np.percentile(v, 2.5)) if v.size else None,
                     "hi": float(np.percentile(v, 97.5)) if v.size else None,
                     "B_effective": int(v.size)}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True, help="query x library distance block (.pkl DataFrame)")
    p.add_argument("--cohort", required=True, help="short cohort tag, e.g. stv, ccdi, comet_selfcheck")
    p.add_argument("--model", default="uni2")
    p.add_argument("--method", default="ot_cosine")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--library-drop-labels", nargs="*", default=None,
                   help="default: the frozen all-tissue protocol, ['MISC']")
    p.add_argument("--query-metadata", default=None)
    p.add_argument("--query-id-col", default="slide_id")
    p.add_argument("--query-label-col", default="tumor_group")
    p.add_argument("--query-patient-col", default=None)
    p.add_argument("--internal-reference-matrix", default=None,
                   help="COMET masked square matrix for the same model/method; "
                        "its label-free statistics are reported beside the cohort's")
    p.add_argument("--iter", type=int, default=PUBLISHED["bootstrap_B"], dest="bootstrap_B")
    p.add_argument("--seed", type=int, default=PUBLISHED["seed"])
    p.add_argument("--bootstrap-mode", default=PUBLISHED["bootstrap_mode"])
    p.add_argument("--task", default=None, help="default external_<cohort>")
    p.add_argument("--assert-known-answer", action="store_true",
                   help="self-check: require C2/G2 known answers (uni2/ot_cosine, COMET as queries)")
    return p


def run(a) -> dict:
    drop = a.library_drop_labels if a.library_drop_labels is not None else \
        list(PUBLISHED_DROP_LABELS[False])
    lib_meta = load_metadata(str(paths.require(paths.slide_metadata(), "slide metadata")),
                             label_col=PUBLISHED["label_col"], id_col=PUBLISHED["id_col"],
                             patient_col=PUBLISHED["patient_col"], drop_labels=drop,
                             min_class_count=None)
    lib_labels = dict(zip(lib_meta[PUBLISHED["id_col"]].astype(str),
                          lib_meta[PUBLISHED["label_col"]].astype(str)))
    lib_patients = dict(zip(lib_meta[PUBLISHED["id_col"]].astype(str),
                            lib_meta[PUBLISHED["patient_col"]].astype(str)))

    M = load_similarity_matrix(str(paths.require(Path(a.matrix), "distance block")))
    M.index = M.index.astype(str)
    M.columns = M.columns.astype(str)
    lib_ids = [c for c in M.columns if c in lib_labels]          # matrix order ([decision record])
    if not lib_ids:
        raise SystemExit("no library columns match the COMET metadata ids")
    labels_order = sorted(set(lib_labels[i] for i in lib_ids))
    lib_class_counts = Counter(lib_labels[i] for i in lib_ids)
    q_ids = [str(i) for i in M.index]

    q_true = None
    q_meta = None
    q_patients = None
    if a.query_metadata:
        qm = pd.read_csv(a.query_metadata, dtype=str)
        qm = qm.dropna(subset=[a.query_id_col, a.query_label_col])
        qm[a.query_id_col] = qm[a.query_id_col].astype(str)
        qm = qm.drop_duplicates(a.query_id_col)
        lookup = dict(zip(qm[a.query_id_col], qm[a.query_label_col].astype(str)))
        q_ids = [q for q in q_ids if q in lookup]
        q_true = [lookup[q] for q in q_ids]
        if a.query_patient_col and a.query_patient_col in qm.columns:
            pl = dict(zip(qm[a.query_id_col], qm[a.query_patient_col].astype(str)))
            q_patients = {q: pl.get(q, q) for q in q_ids}
        else:
            q_patients = {q: q for q in q_ids}      # each query is its own cluster
        q_meta = pd.DataFrame({a.query_id_col: q_ids, a.query_label_col: q_true,
                               "patient": [q_patients[q] for q in q_ids]})
    same_cohort = any(q in lib_labels for q in q_ids)   # self-check mode
    nbr_ids, nbr_labels, nbr_dists, short = rank_block(
        M, lib_ids, lib_labels, lib_patients, q_ids,
        {q: lib_patients.get(q) for q in q_ids} if same_cohort else None, a.depth)

    df = per_query_frame(q_ids, q_true, nbr_ids, nbr_labels, nbr_dists, a.k_values, a.depth)

    results = {"config": {
        "cohort": a.cohort, "model": a.model, "method": a.method,
        "matrix": str(a.matrix), "depth": a.depth, "k_values": list(a.k_values),
        "library": "COMET all-tissue population in matrix column order",
        "library_drop_labels": drop, "n_library": len(lib_ids),
        "library_classes": labels_order, "library_class_counts": dict(lib_class_counts),
        "query_metadata": a.query_metadata, "query_id_col": a.query_id_col,
        "query_label_col": a.query_label_col, "query_patient_col": a.query_patient_col,
        "self_check_mode": same_cohort, "bootstrap_B": a.bootstrap_B, "seed": a.seed,
        "bootstrap_mode": a.bootstrap_mode,
        "protocol": "configs/config.paper.yaml @ 1.0-frozen-20260827",
        "prosper_commit": _git_commit(), "lsf_job": os.environ.get("LSB_JOBID"),
    }, "n_queries": len(q_ids), "queries_with_fewer_than_depth_neighbours": short}

    results["label_free"] = distribution_block(df, a.k_values)

    if a.internal_reference_matrix:
        R = load_similarity_matrix(str(paths.require(Path(a.internal_reference_matrix),
                                                     "internal reference matrix")))
        R.index = R.index.astype(str)
        R.columns = R.columns.astype(str)
        r_lib = [c for c in R.columns if c in lib_labels]
        r_q = [i for i in R.index if i in lib_labels]
        rn_ids, rn_labels, rn_dists, _ = rank_block(
            R, r_lib, lib_labels, lib_patients, r_q,
            {q: lib_patients.get(q) for q in r_q}, a.depth)
        rdf = per_query_frame(r_q, [lib_labels[q] for q in r_q], rn_ids, rn_labels,
                              rn_dists, a.k_values, a.depth)
        results["internal_reference"] = {
            "matrix": str(a.internal_reference_matrix),
            **distribution_block(rdf, a.k_values),
            "note": "COMET queries vs the COMET library, self and same-patient "
                    "excluded; the comparison baseline for the cohort's "
                    "label-free statistics (domain-shift diagnostic)",
        }

    if q_true is not None:
        in_lib = df["true_label"].isin(labels_order)
        d_in = df[in_lib].reset_index(drop=True)
        d_out = df[~in_lib].reset_index(drop=True)
        results["n_in_library_queries"] = int(len(d_in))
        results["n_out_of_library_queries"] = int(len(d_out))
        results["out_of_library"] = {
            "true_label_counts": d_out["true_label"].value_counts().to_dict(),
            **distribution_block(d_out, a.k_values),
            "note": "true label is not a library class: cannot be retrieved by any "
                    "system; reported as behaviour, not scored as error",
        } if len(d_out) else None
        if len(d_in):
            rows = np.array([i for i, q in enumerate(q_ids) if in_lib.iloc[i]], dtype=int)
            true_in = [q_true[i] for i in rows]
            n_rel = np.array([float(lib_class_counts.get(t, 0)) for t in true_in])
            if same_cohort:
                # same class, different patient -- as metrics_cli
                n_rel = np.array([
                    float(sum(1 for j in lib_ids if lib_labels[j] == t
                              and lib_patients.get(j) != lib_patients.get(q)))
                    for q, t in zip([q_ids[i] for i in rows], true_in)])
            point = metrics.compute_all(nbr_labels[rows], true_in,
                                        k_values=a.k_values, n_relevant=n_rel)
            f1s = {}
            per_class_hit = {}
            for k in a.k_values:
                for rule in ("generous", "absolute", "closest"):
                    f1s[f"{rule}_k{k}"] = _f1_block(
                        true_in, d_in[f"pred_{rule}_k{k}"].astype(str).tolist(), labels_order)
                per_class_hit[f"hit_at_{k}"] = {
                    c: float(d_in.loc[d_in["true_label"] == c, f"hit_k{k}"].mean())
                    for c in labels_order if (d_in["true_label"] == c).any()}
            results["point_estimates"] = point
            results["voting_f1"] = f1s
            results["per_class_hit"] = per_class_hit
            results["query_class_counts"] = d_in["true_label"].value_counts().to_dict()
            if a.bootstrap_B > 0:
                qm_in = q_meta[q_meta[a.query_id_col].isin(set(d_in["query"]))]
                results["confidence_intervals"] = bootstrap(
                    d_in, qm_in, a.query_id_col, a.query_label_col, "patient",
                    a.k_values, labels_order, a.bootstrap_B, a.seed, a.bootstrap_mode)

    task = a.task or f"external_{a.cohort}"
    stem = f"{a.cohort}_{a.model}_{a.method}"
    out = paths.result_json(task, f"external_{stem}")
    df.to_csv(out.parent / f"per_query_{stem}.csv", index=False)
    ranked = pd.DataFrame({
        "query": np.repeat(q_ids, a.depth),
        "rank": np.tile(np.arange(1, a.depth + 1), len(q_ids)),
        "neighbour": nbr_ids.reshape(-1), "neighbour_label": nbr_labels.reshape(-1),
        "distance": nbr_dists.reshape(-1)})
    try:
        ranked.to_parquet(out.parent / f"ranked_lists_{stem}.parquet", index=False)
    except Exception:
        ranked.to_csv(out.parent / f"ranked_lists_{stem}.csv.gz", index=False, compression="gzip")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[external] wrote {out}")
    lf = results["label_free"]
    for k in a.k_values:
        print(f"[external] k={k:<3} vote share {lf[f'vote_share_k{k}']['mean']:.3f}  "
              f"entropy {lf[f'vote_entropy_k{k}']['mean']:.3f}  "
              f"strict-ambiguous {lf[f'fraction_ambiguous_strict_k{k}']:.3f}")
    if "point_estimates" in results:
        pe = results["point_estimates"]
        for k in a.k_values:
            print(f"[external] k={k:<3} macro hit@k={pe.get(f'macro_hit_at_{k}'):.4f}  "
                  f"precision@k={pe.get(f'macro_precision_at_{k}'):.4f}  "
                  f"generous macro F1={results['voting_f1'][f'generous_k{k}']['macro_f1']:.4f}")
    return results


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    r = run(a)
    if not a.assert_known_answer:
        return 0
    out_root = paths.output_root()
    ref = json.load(open(out_root / "c2_metric_suite" / "metrics_uni2_ot_cosine.json"))["point_estimates"]
    g2 = json.load(open(out_root / "g2_table_s3" / "slide_ci_uni2_ot_cosine_top5_generous_majority.json"))
    ok = r.get("n_in_library_queries") == 2778
    print(f"[self-check] n_in_library_queries {r.get('n_in_library_queries')} vs 2778 {'OK' if ok else 'MISMATCH'}")
    for k in a.k_values:
        for m in ("macro_hit_at", "macro_precision_at", "micro_hit_at"):
            got, exp = r["point_estimates"][f"{m}_{k}"], ref[f"{m}_{k}"]
            good = abs(got - exp) < 5e-7
            ok &= good
            print(f"[self-check] {m}_{k}: {got:.9f} vs {exp:.9f} {'OK' if good else 'MISMATCH'}")
    got = r["voting_f1"]["generous_k5"]["macro_f1"]
    exp = g2["point_estimates"]["macro_f1"]
    good = abs(got - exp) < 5e-7
    ok &= good
    print(f"[self-check] generous macro F1 k=5: {got:.9f} vs G2 {exp:.9f} {'OK' if good else 'MISMATCH'}")
    print("[self-check] PASS" if ok else "[self-check] FAIL -- do not evaluate any external cohort with this code")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
