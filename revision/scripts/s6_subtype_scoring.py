#!/usr/bin/env python3
"""S6.1-S6.5 scoring driver ([analysis plan], [decision record]; spec:
[internal record] — written and committed before this
ran). Re-scores existing UNI2+OT matrices with subtype labels via the frozen
evaluation machinery imported verbatim from prosper-search/prosper/evaluation
(the code whose known-answer self-check reproduces C2/G2 exactly).

Arms (--arm):
  s61_int_2c / s61_int_3c / s61_int_2c_xeno / s61_fusion   internal RMS
  s61_ext_2c / s61_ext_3c                                  CCDI RMS
  s62 / s62_xeno                                           MYCN
  s64 / s64_ext_hb                                         HB-vs-HCC (+KidRare HB)
  s65_inpc / s65b_gnb                                      NB differentiation
  s65_ext_kidrare                                          within-KidRare NB
  viewa_internal / viewa_ccdi                              deployment addenda
CPU-only; every output goes under $PROSPER_OUT/s6_<family>/.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

CORE = "/PROSPER_WORKSPACE"
sys.path.insert(0, f"{CORE}/prosper-search")
os.environ.setdefault("PROSPER_OUT", f"{CORE}/prosper_revision_output")
OUT = os.environ["PROSPER_OUT"]

from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from prosper.evaluation import metrics, resample  # noqa: E402
from prosper.evaluation.external_cli import (  # noqa: E402
    rank_block, per_query_frame, _f1_block, bootstrap as f1_bootstrap)
from prosper.evaluation.cli import _git_commit  # noqa: E402

MASKED = ("/PROSPER_WORKSPACE/comet-feature-gen/"
          "comet-uni2/output/similarity_matrix/"
          "combined_similarity_matrix_uni2_ot_cosine_tumor_only_no_same_patient.pkl")
FROZEN_META = ("/PROSPER_WORKSPACE/comet-feature-gen/"
               "comet-analysis/data/final_scan_id_disease_map_with_sample_id_patient_id.csv")
CCDI_MATRIX = f"{OUT}/similarity_matrix/combined_similarity_matrix_ccdi_uni2_ot_cosine.pkl"
KIDRARE_MATRIX = f"{OUT}/similarity_matrix/combined_similarity_matrix_kidrare_uni2_ot_cosine.pkl"
KIDRARE_NB_BLOCK = f"{OUT}/s6_5_kidrare_nb/block_kidrare_nb_uni2_ot_cosine.pkl"
INV = f"{OUT}/s6_0_label_inventory"
ALI_CSV = os.environ.get("S6_ALI_CSV", f"{CORE}/prosper_revision/data/ALIdf3Ali.csv")
KS = [1, 3, 5, 10]
B = 5000
SEED = 123


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(task):
    df = pd.read_csv(f"{INV}/manifest_{task}.csv")
    df["slide_id"] = df["slide_id"].astype(str)
    df["patient"] = df["patient_id"].astype(str)
    return df


def load_matrix(path):
    M = pd.read_pickle(path)
    M.index = M.index.astype(str)
    M.columns = M.columns.astype(str)
    return M


def _clusters(meta):
    """Patient index for resampling; a query with no patient id is its own cluster."""
    m = meta.copy()
    m["patient"] = m["patient"].fillna(m["slide_id"])
    return m


def _auroc_enrich(df, nbr_labels, nbr_dists, q_true, q_meta, labels_order,
                  lib_labels, lib_patients, lib_ids, q_patients,
                  positive=None, B_=B, seed=SEED):
    """AUROC from distance-weighted top-5 vote share (binary tasks, spec def)
    + per-class prevalence-mirror enrichment, both with PPS patient CIs."""
    n = len(q_true)
    d5 = nbr_dists[:, :5].astype(float)
    l5 = nbr_labels[:, :5]
    valid = np.isfinite(d5)
    w = np.where(valid, 1.0 / (d5 + 1e-6), 0.0)
    out = {}

    # library prevalence available to each query (excludes own patient)
    lib_lab_arr = np.array([lib_labels[i] for i in lib_ids], dtype=object)
    lib_pat_arr = np.array([lib_patients[i] for i in lib_ids], dtype=object)
    qp = np.array([q_patients[q] for q in df["query"]], dtype=object)
    tl = np.array(q_true, dtype=object)
    avail = (lib_pat_arr[None, :] != qp[:, None])
    # a library slide equal to the query itself is also unavailable
    lid_arr = np.array(lib_ids, dtype=object)
    avail &= (lid_arr[None, :] != np.array(df["query"], dtype=object)[:, None])
    n_avail = avail.sum(1).astype(float)
    prev_true = np.array([
        (avail[i] & (lib_lab_arr == tl[i])).sum() / max(n_avail[i], 1)
        for i in range(n)])
    share_true = np.array([
        float(np.mean([l == tl[i] for l, v in zip(l5[i], valid[i]) if v]))
        if valid[i].any() else np.nan for i in range(n)])
    enrich = share_true - prev_true

    per_class_enrich = {}
    for c in labels_order:
        m = tl == c
        per_class_enrich[c] = {
            "n": int(m.sum()),
            "mean_top5_own_class_share": float(np.nanmean(share_true[m])),
            "mean_prevalence_available": float(np.mean(prev_true[m])),
            "mean_enrichment": float(np.nanmean(enrich[m])),
            "prevalence_random_hit_at_5": float(np.mean(1 - (1 - prev_true[m]) ** 5)),
        }
    out["enrichment"] = per_class_enrich

    if positive is not None:
        pos = (l5 == positive) & valid
        share_w = (w * pos).sum(1) / np.maximum(w.sum(1), 1e-12)
        share_c = pos.sum(1) / np.maximum(valid.sum(1), 1)
        y = (tl == positive).astype(int)
        out["auroc"] = {
            "positive_class": positive,
            "weighted_vote_share_top5": float(roc_auc_score(y, share_w)),
            "count_vote_share_top5": float(roc_auc_score(y, share_c)),
        }
    # patient-clustered CIs
    qm = _clusters(q_meta)
    index = resample.build_patient_index(qm, "slide_id", "label", "patient")
    idx_of = {q: i for i, q in enumerate(df["query"].tolist())}
    rng = np.random.default_rng(seed)
    draws_e = {c: [] for c in labels_order}
    draws_a = []
    for _ in range(B_):
        s = resample.sample_queries(index, labels_order, rng,
                                    mode="pps_cluster_one_slide")
        rows = np.array([idx_of[x] for x in s if x in idx_of], dtype=int)
        if rows.size == 0:
            continue
        rl = tl[rows]
        for c in labels_order:
            sel = rows[rl == c]
            if sel.size:
                draws_e[c].append(float(np.nanmean(enrich[sel])))
        if positive is not None:
            yy = (rl == positive).astype(int)
            if 0 < yy.sum() < yy.size:
                draws_a.append(float(roc_auc_score(yy, share_w[rows])))
    for c in labels_order:
        v = np.array(draws_e[c])
        per_class_enrich[c]["enrichment_ci"] = (
            [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            if v.size else None)
    if positive is not None:
        v = np.array(draws_a)
        out["auroc"]["weighted_ci"] = (
            [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            if v.size else None)
        out["auroc"]["B_effective"] = int(v.size)
    df["top5_own_class_share"] = share_true
    df["available_prevalence_own_class"] = prev_true
    if positive is not None:
        df["weighted_pos_vote_share_top5"] = share_w
    return out


def run_view_b(tag, M, lib, q, out_task, positive=None, depth=10,
               extra_config=None, bootstrap_B=B):
    """View B: queries q against sub-library lib on matrix M (frozen machinery)."""
    lib = lib.dropna(subset=["label"]).drop_duplicates("slide_id")
    q = q.dropna(subset=["label"]).drop_duplicates("slide_id")
    lib_set = set(lib.slide_id)
    lib_ids = [c for c in M.columns if c in lib_set]
    lib_labels = dict(zip(lib.slide_id, lib.label.astype(str)))
    lib_patients = dict(zip(lib.slide_id, lib.patient.astype(str)))
    q_set = set(q.slide_id)
    q_ids = [i for i in M.index if i in q_set]
    qmap_l = dict(zip(q.slide_id, q.label.astype(str)))
    qmap_p = dict(zip(_clusters(q).slide_id, _clusters(q).patient.astype(str)))
    q_true = [qmap_l[i] for i in q_ids]
    q_patients = {i: qmap_p[i] for i in q_ids}
    labels_order = sorted(set(lib_labels[i] for i in lib_ids))

    nbr_ids, nbr_labels, nbr_dists, short = rank_block(
        M, lib_ids, lib_labels, lib_patients, q_ids, q_patients, depth)
    df = per_query_frame(q_ids, q_true, nbr_ids, nbr_labels, nbr_dists, KS, depth)

    lib_lab_arr = np.array([lib_labels[i] for i in lib_ids], dtype=object)
    lib_pat_arr = np.array([lib_patients[i] for i in lib_ids], dtype=object)
    n_rel = np.array([
        float(((lib_lab_arr == t) & (lib_pat_arr != q_patients[qid])
               & (np.array(lib_ids, dtype=object) != qid)).sum())
        for qid, t in zip(q_ids, q_true)])
    point = metrics.compute_all(nbr_labels, q_true, k_values=KS, n_relevant=n_rel)

    f1s, per_class_hit = {}, {}
    for k in KS:
        for rule in ("generous", "absolute", "closest"):
            f1s[f"{rule}_k{k}"] = _f1_block(
                q_true, df[f"pred_{rule}_k{k}"].astype(str).tolist(), labels_order)
        per_class_hit[f"hit_at_{k}"] = {
            c: float(df.loc[df["true_label"] == c, f"hit_k{k}"].mean())
            for c in labels_order if (df["true_label"] == c).any()}

    q_meta = pd.DataFrame({"slide_id": q_ids, "label": q_true,
                           "patient": [q_patients[i] for i in q_ids]})
    cis = f1_bootstrap(df, q_meta, "slide_id", "label", "patient",
                       KS, labels_order, bootstrap_B, SEED, "pps_cluster_one_slide")
    extra = _auroc_enrich(df, nbr_labels, nbr_dists, q_true, q_meta, labels_order,
                          lib_labels, lib_patients, lib_ids, q_patients,
                          positive=positive, B_=bootstrap_B)

    maj = pd.Series([lib_labels[i] for i in lib_ids]).value_counts().idxmax()
    baselines = {
        "majority_class": maj,
        "majority_accuracy": float(np.mean([t == maj for t in q_true])),
        "majority_macro_f1": float(f1_score(q_true, [maj] * len(q_true),
                                            average="macro", labels=labels_order,
                                            zero_division=0)),
    }

    res = {"config": {
        "arm": tag, "view": "B", "depth": depth, "k_values": KS,
        "n_queries": len(q_ids), "n_library": len(lib_ids),
        "labels_order": labels_order,
        "library_class_counts": pd.Series(
            [lib_labels[i] for i in lib_ids]).value_counts().to_dict(),
        "query_class_counts": pd.Series(q_true).value_counts().to_dict(),
        "n_patients": int(pd.Series([q_patients[i] for i in q_ids]).nunique()),
        "bootstrap_B": bootstrap_B, "seed": SEED,
        "bootstrap_mode": "pps_cluster_one_slide",
        "protocol": "configs/config.paper.yaml @ 1.0-frozen-20260827 + s6 spec",
        "prosper_commit": _git_commit(),
        "queries_with_fewer_than_depth_neighbours": int(short),
        **(extra_config or {})},
        "point_estimates": point, "voting_f1": f1s, "per_class_hit": per_class_hit,
        "confidence_intervals": cis, "baselines": baselines, **extra}

    d = f"{OUT}/{out_task}"
    os.makedirs(d, exist_ok=True)
    df.to_csv(f"{d}/per_query_{tag}.csv", index=False)
    ranked = pd.DataFrame({
        "query": np.repeat(q_ids, depth),
        "rank": np.tile(np.arange(1, depth + 1), len(q_ids)),
        "neighbour": nbr_ids.reshape(-1), "neighbour_label": nbr_labels.reshape(-1),
        "distance": nbr_dists.reshape(-1)})
    ranked.to_csv(f"{d}/ranked_lists_{tag}.csv.gz", index=False, compression="gzip")
    with open(f"{d}/result_{tag}.json", "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    g5 = f1s["generous_k5"]
    print(f"[{tag}] n={len(q_ids)} lib={len(lib_ids)} classes={labels_order}")
    print(f"[{tag}] generous k5: acc {g5['accuracy']:.4f} macroF1 {g5['macro_f1']:.4f}  "
          f"closest k5 acc {f1s['closest_k5']['accuracy']:.4f}  "
          f"macro hit@5 {point['macro_hit_at_5']:.4f}")
    if "auroc" in res:
        a = res["auroc"]
        print(f"[{tag}] AUROC({a['positive_class']}) weighted {a['weighted_vote_share_top5']:.4f} "
              f"CI {a.get('weighted_ci')}")
    return res


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

def internal_pop(task, drop_xeno=True, drop_patients=(), keep_labels=None):
    man = load_manifest(task)
    if drop_xeno:
        man = man[man["sample_type"] != "Xenograft"]
    if drop_patients:
        man = man[~man["patient"].isin(set(drop_patients))]
    if keep_labels:
        man = man[man["label"].isin(set(keep_labels))]
    return man[["slide_id", "label", "patient", "met", "sample_type"]].copy()


def fusion_pop():
    """RMS non-xeno slides with a normalised fusion status from ALIdf3Ali."""
    man = internal_pop("s61_rms_comet")
    master = pd.read_csv(ALI_CSV, low_memory=False)
    master["slide_key"] = (master["Ali_Slides"].astype(str)
                           .str.replace(r"\.svs$", "", regex=True))
    cols = ["PA3.fusion", "PA7.fusion", "FKHR..FOO1..FISH",
            "PA3..FOO1.PCR", "PA7..FOO1.PCR"]

    def norm(v):
        if not isinstance(v, str):
            return None
        s = v.strip().lstrip("*").upper()
        if s.startswith("POS"):
            return "POS"
        if s.startswith("NEG"):
            return "NEG"
        return None

    rows = {}
    for _, r in master.iterrows():
        sid = r["slide_key"]
        if sid in (None, "nan", ""):
            continue
        vals = [norm(r[c]) for c in cols]
        vals = [v for v in vals if v]
        if not vals:
            continue
        status = "fusion_positive" if "POS" in vals else "fusion_negative"
        # multiple master rows per slide: any POS wins; record conflict count
        prev = rows.get(sid)
        rows[sid] = status if prev is None else (
            "fusion_positive" if "fusion_positive" in (prev, status) else prev)
    man = man[man["slide_id"].isin(rows)].copy()
    man["histology"] = man["label"]
    man["label"] = man["slide_id"].map(rows)
    return man


# ---------------------------------------------------------------------------
# Special arms
# ---------------------------------------------------------------------------

def s64_ext_hb():
    """KidRare HB queries vs COMET LIVER sub-library: one-sided concordance."""
    M = load_matrix(KIDRARE_MATRIX)
    lib = internal_pop("s64_hbhcc_comet")
    km = pd.read_csv(f"{OUT}/external_kidrare/manifest_inscope.csv")
    km["slide_id"] = km["slide_id"].astype(str)
    hb = km[km.tumor_group == "LIVER"]
    q_ids = [i for i in M.index if i in set(hb.slide_id)]
    lib_ids = [c for c in M.columns if c in set(lib.slide_id)]
    lib_labels = dict(zip(lib.slide_id, lib.label))
    lib_patients = dict(zip(lib.slide_id, lib.patient))
    nbr_ids, nbr_labels, nbr_dists, short = rank_block(
        M, lib_ids, lib_labels, lib_patients, q_ids, None, 10)
    hb_share5 = (nbr_labels[:, :5] == "HB").mean(1)
    top1_hb = (nbr_labels[:, 0] == "HB").astype(float)
    prev = float(np.mean([lib_labels[i] == "HB" for i in lib_ids]))
    rng = np.random.default_rng(SEED)
    n = len(q_ids)
    draws = np.array([hb_share5[rng.integers(0, n, n)].mean() for _ in range(B)])
    res = {"config": {"arm": "s64_ext_hb", "n_queries": n, "n_library": len(lib_ids),
                      "library_hb_prevalence": prev, "matrix": KIDRARE_MATRIX,
                      "bootstrap": "slide-level (KidRare has no patient IDs)",
                      "prosper_commit": _git_commit()},
           "mean_top5_hb_share": float(hb_share5.mean()),
           "mean_top5_hb_share_ci": [float(np.percentile(draws, 2.5)),
                                     float(np.percentile(draws, 97.5))],
           "top1_hb_rate": float(top1_hb.mean()),
           "enrichment_over_prevalence": float(hb_share5.mean() - prev),
           "note": ("one-sided concordance check: no external HCC cohort exists; "
                    "queries are all KidRare HB")}
    d = f"{OUT}/s6_4_hbhcc"
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({"query": q_ids, "top5_hb_share": hb_share5,
                  "top1_hb": top1_hb}).to_csv(f"{d}/per_query_s64_ext_hb.csv",
                                              index=False)
    with open(f"{d}/result_s64_ext_hb.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[s64_ext_hb] n={n} mean top-5 HB share {res['mean_top5_hb_share']:.4f} "
          f"CI {res['mean_top5_hb_share_ci']} vs prevalence {prev:.4f}")
    return res


def viewa_internal():
    """Deployment addendum: subtype vote over first <=5 labelled neighbours in the
    unrestricted tumour-only library (depth 20), + same-group routing."""
    M = load_matrix(MASKED)
    frozen = pd.read_csv(FROZEN_META)
    frozen["slide_id"] = frozen["Slide.Scan.File"].astype(str)
    frozen = frozen[~frozen["Tumor.group"].isin(["MISC", "NORMAL"])]
    group = dict(zip(frozen["slide_id"], frozen["Tumor.group"].astype(str)))
    gpat = dict(zip(frozen["slide_id"], frozen["UID.Subject"].astype(str)))
    lib_ids = [c for c in M.columns if c in group]
    out = {}
    arms = {
        "s61_rms": internal_pop("s61_rms_comet", keep_labels=["ERMS", "ARMS"]),
        "s62_mycn": internal_pop("s62_mycn_comet", drop_patients=["SJHBB3FZN"]),
        "s64_hbhcc": internal_pop("s64_hbhcc_comet"),
        "s65_inpc": internal_pop("s65_inpc_comet"),
        "s65b_gnb": internal_pop("s65b_gnb_comet"),
    }
    for tag, pop in arms.items():
        sub_label = dict(zip(pop.slide_id, pop.label))
        q_ids = [i for i in M.index if i in sub_label]
        q_pat = {i: gpat[i] for i in q_ids}
        nbr_ids, nbr_labels, nbr_dists, _ = rank_block(
            M, lib_ids, group, gpat, q_ids, q_pat, 20)
        routing, correct_gen, correct_close, coverage = [], [], [], []
        for i, qid in enumerate(q_ids):
            qg = group[qid]
            top5g = [group.get(n) for n in nbr_ids[i, :5] if n is not None]
            routing.append(np.mean([g == qg for g in top5g]) if top5g else np.nan)
            labs = [sub_label[n] for n in nbr_ids[i] if n in sub_label][:5]
            coverage.append(len(labs))
            if labs:
                counts = pd.Series(labs).value_counts()
                tied = counts[counts == counts.max()].index.tolist()
                t = sub_label[qid]
                correct_gen.append(float(t in tied))
                correct_close.append(float(next(l for l in labs if l in tied) == t))
            else:
                correct_gen.append(np.nan)
                correct_close.append(np.nan)
        out[tag] = {
            "n_queries": len(q_ids),
            "mean_top5_same_group_routing": float(np.nanmean(routing)),
            "mean_n_labelled_neighbours_of_5": float(np.mean(coverage)),
            "frac_queries_with_5_labelled": float(np.mean(np.array(coverage) == 5)),
            "generous_accuracy_first5_labelled": float(np.nanmean(correct_gen)),
            "closest_accuracy_first5_labelled": float(np.nanmean(correct_close)),
        }
        print(f"[viewa:{tag}] routing {out[tag]['mean_top5_same_group_routing']:.3f} "
              f"generous acc {out[tag]['generous_accuracy_first5_labelled']:.4f} "
              f"(labelled/5 = {out[tag]['mean_n_labelled_neighbours_of_5']:.2f})")
    d = f"{OUT}/s6_viewa"
    os.makedirs(d, exist_ok=True)
    res = {"config": {"view": "A", "library": "tumour-only population minus MISC/NORMAL",
                      "n_library": len(lib_ids), "depth": 20,
                      "rule": "vote over first <=5 subtype-labelled neighbours",
                      "prosper_commit": _git_commit()}, "arms": out}
    with open(f"{d}/result_viewa_internal.json", "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def viewa_ccdi():
    """CCDI deployment addendum from the existing all-tissue ranked lists."""
    rl = pd.read_csv(f"{OUT}/external_ccdi_inscope/ranked_lists_ccdi_uni2_ot_cosine.csv.gz")
    rl["query"] = rl["query"].astype(str)
    rl["neighbour"] = rl["neighbour"].astype(str)
    q = load_manifest("s61_rms_ccdi")
    pop = internal_pop("s61_rms_comet")
    sub_label = dict(zip(pop.slide_id, pop.label))
    qlab = dict(zip(q.slide_id, q.label))
    stats = []
    for qid, g in rl.groupby("query", sort=False):
        if qid not in qlab:
            continue
        g = g.sort_values("rank")
        rms_route = np.mean(g["neighbour_label"].head(5) == "RMS")
        labs = [sub_label[n] for n in g["neighbour"] if n in sub_label][:5]
        row = {"query": qid, "true_label": qlab[qid], "top5_rms_routing": rms_route,
               "n_labelled": len(labs)}
        if labs:
            counts = pd.Series(labs).value_counts()
            tied = counts[counts == counts.max()].index.tolist()
            row["generous_correct"] = float(qlab[qid] in tied)
            row["closest_correct"] = float(
                next(l for l in labs if l in tied) == qlab[qid])
        stats.append(row)
    df = pd.DataFrame(stats)
    res = {"config": {"view": "A", "source": "existing C3 ranked lists (depth 10, "
                      "all-tissue library)", "rule": "vote over first <=5 "
                      "subtype-labelled RMS neighbours", "prosper_commit": _git_commit()},
           "n_queries": int(len(df)),
           "mean_top5_rms_routing": float(df["top5_rms_routing"].mean()),
           "mean_n_labelled_of_5": float(df["n_labelled"].mean()),
           "generous_accuracy": float(df["generous_correct"].mean()),
           "closest_accuracy": float(df["closest_correct"].mean()),
           "by_class_generous": df.groupby("true_label")["generous_correct"]
                                  .mean().to_dict()}
    d = f"{OUT}/s6_viewa"
    os.makedirs(d, exist_ok=True)
    df.to_csv(f"{d}/per_query_viewa_ccdi.csv", index=False)
    with open(f"{d}/result_viewa_ccdi.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[viewa:ccdi] n={len(df)} routing {res['mean_top5_rms_routing']:.3f} "
          f"generous acc {res['generous_accuracy']:.4f}")
    return res


def s65_ext_kidrare():
    """Within-KidRare NB View B on the labelled 136, from the new OT block."""
    M = load_matrix(KIDRARE_NB_BLOCK)
    km = pd.read_csv(f"{OUT}/external_kidrare/manifest_inscope.csv")
    km["slide_id"] = km["slide_id"].astype(str)
    nb = km[(km.tumor_group == "NB") & km["keep_subtype"].notna()].copy()
    short_map = {"Poorly differentiated neuroblastoma": "poorly_differentiated",
                 "Differentiating neuroblastoma": "differentiating",
                 "Ganglioneuroblastoma, intermixed": "gnb_intermixed"}
    nb["label"] = nb["keep_subtype"].map(short_map)
    nb["patient"] = nb["slide_id"]  # no patient IDs: slide-level clusters
    pop = nb[["slide_id", "label", "patient"]].copy()
    pop["met"] = None
    pop["sample_type"] = "Patient tumor"
    extra = {"matrix": KIDRARE_NB_BLOCK,
             "bootstrap_note": "no patient IDs in KidRare: slide-level clusters",
             "labels": "keep_subtype (spec C13 amendment archive)"}
    run_view_b("s65_ext_3c", M, pop, pop, "s6_5_kidrare_nb", extra_config=extra)
    binp = pop[pop.label.isin(["poorly_differentiated", "differentiating"])]
    run_view_b("s65_ext_binary", M, binp, binp, "s6_5_kidrare_nb",
               positive="poorly_differentiated", extra_config=extra)


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True)
    a = p.parse_args(argv)
    arm = a.arm

    if arm.startswith("s61_int") or arm in ("s62", "s62_xeno", "s64", "s65_inpc",
                                            "s65b_gnb", "s61_fusion"):
        M = load_matrix(MASKED)

    if arm == "s61_int_2c":
        pop = internal_pop("s61_rms_comet", keep_labels=["ERMS", "ARMS"])
        run_view_b(arm, M, pop, pop, "s6_1_rms", positive="ARMS",
                   extra_config={"matrix": MASKED, "xenograft": "excluded"})
    elif arm == "s61_int_3c":
        pop = internal_pop("s61_rms_comet")
        run_view_b(arm, M, pop, pop, "s6_1_rms",
                   extra_config={"matrix": MASKED, "xenograft": "excluded",
                                 "note": "exploratory 3-class row incl. Spindle"})
    elif arm == "s61_int_2c_xeno":
        pop = internal_pop("s61_rms_comet", drop_xeno=False,
                           keep_labels=["ERMS", "ARMS"])
        run_view_b(arm, M, pop, pop, "s6_1_rms", positive="ARMS",
                   extra_config={"matrix": MASKED,
                                 "xenograft": "included (sensitivity)"})
    elif arm == "s61_fusion":
        pop = fusion_pop()
        xt = pd.crosstab(pop["histology"], pop["label"])
        print("[s61_fusion] histology x fusion:\n", xt)
        run_view_b(arm, M, pop, pop, "s6_1_rms", positive="fusion_positive",
                   extra_config={"matrix": MASKED, "xenograft": "excluded",
                                 "histology_vs_fusion": xt.to_dict(),
                                 "rule": "fusion_positive = any assay POS"})
    elif arm == "s61_ext_2c":
        Mc = load_matrix(CCDI_MATRIX)
        lib = internal_pop("s61_rms_comet", keep_labels=["ERMS", "ARMS"])
        q = load_manifest("s61_rms_ccdi")
        q = q[q.label.isin(["ERMS", "ARMS"])]
        run_view_b(arm, Mc, lib, q, "s6_1_rms", positive="ARMS",
                   extra_config={"matrix": CCDI_MATRIX,
                                 "library": "COMET RMS non-xeno (histologic)",
                                 "queries": "CCDI RMS post-SW1, participant clusters"})
    elif arm == "s61_ext_3c":
        Mc = load_matrix(CCDI_MATRIX)
        lib = internal_pop("s61_rms_comet")
        q = load_manifest("s61_rms_ccdi")
        run_view_b(arm, Mc, lib, q, "s6_1_rms",
                   extra_config={"matrix": CCDI_MATRIX,
                                 "library": "COMET RMS non-xeno (histologic)",
                                 "queries": "CCDI RMS post-SW1, participant clusters"})
    elif arm == "s62":
        pop = internal_pop("s62_mycn_comet", drop_patients=["SJHBB3FZN"])
        run_view_b(arm, M, pop, pop, "s6_2_mycn", positive="amplified",
                   extra_config={"matrix": MASKED, "xenograft": "excluded",
                                 "excluded_patient": "SJHBB3FZN (discordant labels)"})
    elif arm == "s62_xeno":
        pop = internal_pop("s62_mycn_comet", drop_xeno=False)
        run_view_b(arm, M, pop, pop, "s6_2_mycn", positive="amplified",
                   extra_config={"matrix": MASKED,
                                 "xenograft": "included (sensitivity), "
                                              "SJHBB3FZN also included"})
    elif arm == "s64":
        pop = internal_pop("s64_hbhcc_comet")
        run_view_b(arm, M, pop, pop, "s6_4_hbhcc", positive="HCC",
                   extra_config={"matrix": MASKED, "xenograft": "excluded"})
    elif arm == "s64_ext_hb":
        s64_ext_hb()
    elif arm == "s65_inpc":
        pop = internal_pop("s65_inpc_comet")
        run_view_b(arm, M, pop, pop, "s6_5_nbdiff", positive="unfavorable",
                   extra_config={"matrix": MASKED, "xenograft": "excluded"})
    elif arm == "s65b_gnb":
        pop = internal_pop("s65b_gnb_comet")
        pop["label"] = pop["label"].replace({"Ganglioneuroblastoma": "GNB"})
        run_view_b(arm, M, pop, pop, "s6_5_nbdiff", positive="GNB",
                   extra_config={"matrix": MASKED, "xenograft": "excluded"})
    elif arm == "s65_ext_kidrare":
        s65_ext_kidrare()
    elif arm == "viewa_internal":
        viewa_internal()
    elif arm == "viewa_ccdi":
        viewa_ccdi()
    else:
        raise SystemExit(f"unknown arm {arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
